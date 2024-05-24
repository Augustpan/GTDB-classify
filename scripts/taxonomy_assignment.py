'''
@license: (C) Copyright 2024, Augustpan
@author: Augustpan
@email: yfpan21@m.fudan.edu.cn
@tel: 135****9152
@datetime: 2024/5/24 16:40
@project: GTDB-classify
@file: taxonomy_assignment.py
@desc: taxonomy assignment tool for bacterial and archaeal marker genes.
'''

import Bio
from Bio import Phylo
import numpy as np
import pandas as pd
import sys
import json
from multiprocessing import Pool
from functools import partial
from copy import deepcopy
from tqdm import tqdm

def calculate_red(tree):
    # Cache for storing precomputed distances
    distance_cache = {}
    extant_taxa_cache = {}

    def get_extant_taxa(node):
        """Get all extant taxa (leaf nodes) under a given node."""
        if node.is_terminal():
            return [node]
        extant_taxa = []
        stack = [node]
        while stack:
            current_node = stack.pop()
            if current_node.is_terminal():
                extant_taxa.append(current_node)
            else:
                stack.extend(current_node.clades)
        return extant_taxa

    def get_extant_taxa2(node):
        """Get all extant taxa (leaf nodes) under a given node."""
        if node in extant_taxa_cache:
            return extant_taxa_cache[node]
        if node.is_terminal():
            return 1
        return np.sum([get_extant_taxa2(child) for child in node.clades])

    def calculate_u(node, bias):
        """Calculate the average branch length from the node to all extant taxa descendant from it."""
        extant_taxa = get_extant_taxa(node)
        branch_lengths = [tree.distance(node, taxon) + bias for taxon in extant_taxa]
        return np.mean(branch_lengths)

    def calculate_u2(node, bias):
        if node in distance_cache:
            return distance_cache[node]
        if node.is_terminal():
            return bias

        a = np.sum([(calculate_u2(child, child.branch_length) + bias) * get_extant_taxa2(child) for child in node.clades])
        b = np.sum([get_extant_taxa2(child) for child in node.clades])
        average_length = a / b
        distance_cache[node] = average_length

        return average_length

    def assign_red(node, parent_red=0):
        """Assign RED values to each node in the tree."""
        stack = [(node, parent_red)]
        while stack:
            current_node, current_red = stack.pop()
            if current_node.is_terminal():
                current_node.red = 1.0
            elif current_node == tree.root:
                current_node.red = 0.0
            else:
                d = current_node.branch_length
                #u = calculate_u(current_node, d)
                u = calculate_u2(current_node, d)
                #assert u-u2 < np.finfo(float).eps
                current_node.red = current_red + (d / u) * (1 - current_red)
            for child in current_node.clades:
                stack.append((child, current_node.red))

    # Initialize RED calculation from the root of the tree
    assign_red(tree.root)

def write_tree_with_red(tree, output_file):
    # Calculate RED values and assign them to each node
    red_values = calculate_red(tree)

    # Add RED values to the node names
    for node in tree.find_clades():
        if node.is_terminal():
            pass
        elif node == tree.root:
            pass
        else:
            if node.confidence:
                node.name = f"RED={node.red:.4f};BOOTSTRAP={node.confidence:.4f}"
            else:
                node.name = f"RED={node.red:.4f}"

            node.confidence = None

    # Write the tree with RED values to a new Newick file
    Phylo.write(tree, output_file, "newick")

def load_taxonomy(filename):
    taxonomy = pd.read_csv(filename, delimiter="\t", header=None)
    taxonomy.columns = ["genome_acc", "taxonomy"]

    rank_names = ["domain", "phylum", "class", "order", "family", "genus", "species"]
    rank_dict = {rank: [] for rank in rank_names}

    for ind, row in taxonomy.iterrows():
        rank_info = row.taxonomy.split(";")
        for i, rank in enumerate(rank_names):
            rank_dict[rank].append(rank_info[i])

    rank_table = pd.DataFrame(rank_dict)

    return pd.concat([taxonomy, rank_table], axis=1).drop("taxonomy", axis=1)

def check_monophyleticity(df, tree, remove = []):
    tree_ref = deepcopy(tree)

    for node_name in remove:
        try:
            tree_ref.prune(node_name)
        except:
            pass

    node_list = [node for node in tree_ref.find_clades(terminal = True) if node.name in list(df.genome_acc)]
    result = tree_ref.is_monophyletic(node_list)
    if result:
        return True, len(node_list)
    return False, len(node_list)

def tree_search_dijkstra(
    tree: Bio.Phylo.BaseTree.Tree,
    query: Bio.Phylo.BaseTree.Clade,
    is_target: callable
):
    path = tree.get_path(query)
    path.reverse()

    p_queue = [(path[0], 0)]
    for i in range(1, len(path)):
        dist = p_queue[i-1][1] + path[i-1].branch_length
        p_queue.append((path[i], dist))
    p_queue.append((tree.root, p_queue[-1][1] + path[-1].branch_length))
    visited = {p_queue.pop(0)}

    while p_queue:
        node, dist = p_queue.pop(0)
        if node in visited:
            continue
        if is_target(node):
            return node, dist
        visited.add(node)
        for child in node.clades:
            if child not in visited:
                p_queue.append((child, dist + child.branch_length))
        p_queue.sort(key=lambda x: x[1])

def is_nested(tree, clades_a, clades_b):
    mrca_a = tree.common_ancestor(clades_a)
    mrca_b = tree.common_ancestor(clades_b)
    return mrca_a.is_parent_of(mrca_b)

if __name__ == "__main__":

    #####################################################
    tree_file = "test-large.nwk"
    redtree_file = "test-large-with-red.nwk"
    taxonomy_file = "bac120_taxonomy_r214_reps.tsv"
    monophyletic_clades_file = "monophyletic_clades.json"
    red_file = "red.json"
    load_monophyletic_clades_from_file = False
    do_mid_point_root = False
    nproc = 72
    ######################################################

    # Load the tree from a Newick file
    tree = Phylo.read(tree_file, "newick")
    print("Tree loaded.", file=sys.stderr)

    do_mid_point_root = False
    if do_mid_point_root:
        tree.root_at_midpoint()
        print("Mid-point-rooted.", file=sys.stderr)
    else:
        print("Skip mid-point-rooting.", file=sys.stderr)

    # Calculate RED values
    calculate_red(tree)
    print("RED calculation finished.", file=sys.stderr)

    if redtree_file:
        write_tree_with_red(tree, redtree_file)

    tax_table = load_taxonomy(taxonomy_file)
    print("Taxonomy loaded.", file=sys.stderr)

    tip_labels = [node.name for node in tree.find_clades(terminal = True)]
    tax_table_sub = tax_table[tax_table.genome_acc.isin(tip_labels)]
    reference_names = set(tax_table_sub.genome_acc)
    query_names = set(tip_labels) - reference_names

    tree_ref = deepcopy(tree)
    for x in query_names:
        tree_ref.prune(x)
    node_list_ref = list(tree_ref.find_clades(terminal = True))
    node_name_ref = [node.name for node in node_list_ref]
    node_df_ref = pd.DataFrame({"node":node_list_ref, "genome_acc":node_name_ref})

    node_list_tree = list(tree.find_clades(terminal = True))
    node_name_tree = [node.name for node in node_list_tree]
    node_df = pd.DataFrame({"node":node_list_tree, "genome_acc":node_name_tree})

    def worker(taxon, rank):
        df = tax_table_sub[tax_table_sub[rank] == taxon]

        #node_list = [node for node in node_list_ref if node.name in set(df.genome_acc)]
        node_list = list(node_df_ref[node_df_ref.genome_acc.isin(df.genome_acc)].node)

        num_taxa = len(node_list)
        mrca = None
        if num_taxa > 1:
            mrca = tree_ref.is_monophyletic(node_list)
            if mrca:
                is_mono = True
            else:
                is_mono = False
        else:
            is_mono = True
        return is_mono, num_taxa, taxon, mrca

    rank_names = ["domain", "phylum", "class", "order", "family", "genus", "species"]

    if load_monophyletic_clades_from_file:
        with open(monophyletic_clades_file) as f:
            monophyletic_clades = json.load(f)
        with open(red_file) as f:
            red_table = json.load(f)
        print("Monophyleticity file loaded.", file=sys.stderr)
    else:
        monophyletic_clades = {rank: [] for rank in rank_names}
        red_table = {rank: [] for rank in rank_names}
        with Pool(processes=nproc) as pool:
            for rank in rank_names:
                print(f"Assessing monophyleticity: {rank}", file=sys.stderr)
                results = pool.map(partial(worker, rank=rank), set(tax_table_sub[rank]))
                for is_mono, num_taxa, taxon, mrca in results:
                    if is_mono and num_taxa > 1:
                        red_table[rank].append(mrca.red)
                        monophyletic_clades[rank].append(taxon)
        with open(monophyletic_clades_file, "w") as f:
            json.dump(monophyletic_clades, f)
        with open(red_file, "w") as f:
            json.dump(red_table, f)

        print("Monophyleticity check done.", file=sys.stderr)
        print(f"# Reference RED stat:")
        for rank in red_table:
            if red_table[rank]:
                q75, q50, q25 = np.percentile(red_table[rank], [75, 50, 25])
            else:
                q75, q50, q25 = np.nan, np.nan, np.nan
            print(f"{rank}\t{q50}\t{q75}\t{q25}")

    print("Building MRCA map.", file=sys.stderr)
    mrca_map = {}
    for rank in monophyletic_clades:
        for taxon in monophyletic_clades[rank]:
            selected_taxa = list(tax_table_sub[tax_table_sub[rank] == taxon].genome_acc)
            node_list = list(node_df[node_df.genome_acc.isin(selected_taxa)].node)
            mrca = tree.common_ancestor(node_list)
            mrca_map[taxon] = mrca

    print("Start Taxonomy assignments.", file=sys.stderr)
    print("# Taxonomy assignments.")
    i = 0
    for name in tqdm(query_names):
        i += 1
        node = tree.find_any(name)
        taxon_assigned = False
        cloest_ref, dist = tree_search_dijkstra(tree, node, lambda x: x.is_terminal() and x.name in reference_names)
        ranks = list(tax_table_sub[tax_table_sub.genome_acc == cloest_ref.name].iloc[0,1:8])
        for rank, taxon in reversed(list(zip(rank_names, ranks))):
            if taxon in monophyletic_clades[rank]:
                if mrca_map:
                    mrca = mrca_map[taxon]
                    if mrca.is_parent_of(node):
                        print(f"{node.name}\t{rank}\t{taxon}")
                        taxon_assigned = True
                        break
                else:
                    selected_taxa = list(tax_table_sub[tax_table_sub[rank] == taxon].genome_acc)
                    clades_a = list(node_df[node_df.genome_acc.isin(selected_taxa)].node)
                    clades_b = [node]
                    if is_nested(tree, clades_a, clades_b):
                        print(f"{node.name}\t{rank}\t{taxon}")
                        taxon_assigned = True
                        break

        if not taxon_assigned:
            print(f"{node.name}\t\t")

    print("Start RED assignment.", file=sys.stderr)
    print("# RED value for MRCA of pairs of queries")
    for name in tqdm(query_names):
        node = tree.find_any(name)

        cloest_node, dist = tree_search_dijkstra(tree, node, lambda x: x.is_terminal() and x.name != node.name)
        mrca = tree.common_ancestor([node, cloest_node])

        cloest_ref, dist = tree_search_dijkstra(tree, node, lambda x: x.is_terminal() and x.name in reference_names)
        mrca_ref = tree.common_ancestor([node, cloest_ref])

        print(f"{node.name}\t{cloest_ref.name}\t{mrca_ref.red}\t{cloest_node.name}\t{mrca.red}")
