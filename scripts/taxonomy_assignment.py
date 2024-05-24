'''
@license: (C) Copyright 2024, Augustpan
@author: Augustpan
@email: yfpan21@m.fudan.edu.cn
@tel: 135****9152
@datetime: 2024/5/24 12:11
@project: GTDB-classify
@file: taxonomy_assignment.py
@desc: taxonomy assignment tool for bacterial and archaeal marker genes.
'''

import Bio
from Bio import Phylo
import numpy as np
import pandas as pd

from multiprocessing import Pool
from functools import partial
from copy import deepcopy

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
    # Load the tree from a Newick file
    tree = Phylo.read("input_tree.nwk", "newick")
    print("Tree loaded.")

    #tree.root_at_midpoint()
    #print("Mid-point-rooted.")

    # Calculate RED values
    calculate_red(tree)
    print("RED calculation finished.")

    #tax_table = load_taxonomy("ar53_taxonomy_r214_reps.tsv")
    tax_table = load_taxonomy("bac120_taxonomy_r214_reps.tsv")
    print("Taxonomy loaded.")

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
    monophyletic_clades = {rank: [] for rank in rank_names}

    red_table = {rank: [] for rank in rank_names}
    with Pool(processes=12) as pool:
        for rank in rank_names:
            print(f"Assessing monophyleticity: {rank}")
            results = pool.map(partial(worker, rank=rank), set(tax_table_sub[rank]))
            for is_mono, num_taxa, taxon, mrca in results:
                if is_mono and num_taxa > 1:
                    red_table[rank].append(mrca.red)
                    monophyletic_clades[rank].append(taxon)

    print(f"# Reference RED stat:")
    for rank in red_table:
        q75, q25 = np.percentile(red_table[rank], [75 ,25])
        print(f"{rank}\t{np.median(red_table[rank])}\t{q75}\t{q25}")

    print("# Taxonomy assignments.")
    for node in tree.find_clades(terminal = True):
        if node.name not in query_names:
            continue

        taxon_assigned = False
        cloest_ref, dist = tree_search_dijkstra(tree, node, lambda x: x.is_terminal() and x.name in reference_names)
        ranks = list(tax_table_sub[tax_table_sub.genome_acc == cloest_ref.name].iloc[0,1:8])
        for rank, taxon in reversed(list(zip(rank_names, ranks))):
            if taxon in monophyletic_clades[rank]:
                tree_tmp = deepcopy(tree)

                selected_taxa = list(tax_table_sub[tax_table_sub[rank] == taxon].genome_acc)
                selected_taxa.append(node.name)
                node_list = [x for x in tree_tmp.find_clades(terminal = True) if x.name in selected_taxa]

                for name in query_names:
                    if name != node.name:
                        tree_tmp.prune(name)

                if tree_tmp.is_monophyletic(node_list):
                    selected_taxa = list(tax_table_sub[tax_table_sub[rank] == taxon].genome_acc)
                    clades_a = [x for x in tree_tmp.find_clades(terminal = True) if x.name in selected_taxa]
                    clades_b = list(tree_tmp.find_clades(terminal = True, name = node.name))
                    if is_nested(tree_tmp, clades_a, clades_b):
                        print(f"{node.name}\t{rank}\t{taxon}")
                        taxon_assigned = True
                        break
        if not taxon_assigned:
            print(f"{node.name}\t\t")

    print("# RED value for MRCA of pairs of queries")
    for node in tree.find_clades(terminal = True):
        if node.name not in query_names:
            continue
        cloest_ref, dist = tree_search_dijkstra(tree, node, lambda x: x.is_terminal() and x.name in query_names and x.name != node.name)
        mrca = tree.common_ancestor([node, cloest_ref])
        print(f"{node.name}\t{cloest_ref.name}\t{mrca.red}")
