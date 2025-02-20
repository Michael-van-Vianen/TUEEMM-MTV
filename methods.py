import copy
import pandas as pd
import numpy as np
import random as rand
import tqdm
import networkx as nx
import pickle as pkl
import gower
import math
import matplotlib.pyplot as plt

from concurrent.futures import ProcessPoolExecutor, as_completed


def topK(groups, k):
    top_k = copy.deepcopy(groups)
    i = 0
    while i < len(top_k) and i < k:
        top_group = top_k.iloc[i]
        ref_group = set(top_group['reference'])
        sub_group = set(top_group['subgroup'])

        rows_to_drop = []
        for idx, row in top_k.iloc[i + 1:].iterrows():
            overlap_ref = len(ref_group.intersection(row['reference'])) / (len(ref_group) if len(ref_group) else 1)
            overlap_sub = len(sub_group.intersection(row['subgroup'])) / (len(sub_group) if len(sub_group) else 1)
            if overlap_ref > 0.75 and overlap_sub > 0.75:
                rows_to_drop.append(idx)

        top_k = top_k.drop(rows_to_drop)
        i += 1

    return top_k.iloc[:k]


def getAttributes(x, lu):
    numeric_cols = lu.select_dtypes(include=[np.number]).columns
    return [float(i) for i in lu.loc[x, numeric_cols].tolist()]


def ranking(x, S, lu):
    D_dict = nx.shortest_path_length(S, x)
    distances = list(D_dict.values())
    D = list(zip(D_dict.keys(), D_dict.values()))

    for d in np.unique(distances):
        left = distances.index(d)
        right = len(distances) - distances[::-1].index(d)
        tielist = D[left:right]

        gowerlist = gower.gower_matrix(
            pd.DataFrame([getAttributes(x, lu)]),
            pd.DataFrame([getAttributes(t[0], lu) for t in tielist])
        )
        sorted_gower = [node for _, node in sorted(zip(gowerlist[0], tielist))]
        D[left:right] = sorted_gower
    ranks = [(node[0], lu.loc[node[0]]['target']) for node in D]
    return ranks


def Q(S, G, target):
    S_size = len(S)
    G_size = len(G)
    if S_size == 0 or G_size == 0:
        return 0

    cover = S_size / G_size
    n_target_S = sum([x[1] for x in S])
    n_target_G = sum([x[1] for x in G])

    WRAcc = (cover ** 0.5) * ((n_target_S / S_size) - (n_target_G / G_size))
    return abs(WRAcc)


def Q2(S, G, target):
    if len(S) == 0 or len(G) == 0:
        return 0

    cover = len(S) / len(G)
    target_counts_S = sum(x[1] for x in S)
    target_counts_G = sum(x[1] for x in G)
    ratio_s = target_counts_S / len(S)
    ratio_g = target_counts_G / len(G)

    arr_s = np.array([ratio_s, 1 - ratio_s])
    arr_g = np.array([ratio_g, 1 - ratio_g])

    non_zero_mask = (arr_s > 0)
    KL_divergence = np.sum(arr_s[non_zero_mask] * np.log(arr_s[non_zero_mask] / arr_g[non_zero_mask]))

    WKL = cover * KL_divergence
    return WKL


def QTest(S, G, target):
    return rand.uniform(0, 1)


def Discovery(ranks, G, lu, ablation_mode=False):
    G_list_target = list(zip(list(G.nodes), [lu.loc[x]['target'] for x in list(G.nodes)]))

    if not ablation_mode:
        rho = 0
        sigma = 0
        best = 0

        tempranks = [x for x in ranks[0:4]]
        for i in range(5, len(ranks) + 1):
            tempranks.append(ranks[i - 1])
            q = Q(tempranks, G_list_target, 'target')
            if q >= best:
                best = q
                rho = i

        best = 0
        tempranks = [x for x in ranks[0:2]]
        for i in range(3, rho + 1):
            tempranks.append(ranks[i - 1])
            q = Q(tempranks, [x for x in ranks[0:rho]], 'target')
            if q >= best:
                best = q
                sigma = i
        return rho, sigma, best, ranks

    else:
        rho = len(ranks)
        best = 0
        sigma = 0
        tempranks = [x for x in ranks[0:4]]
        for i in range(5, rho + 1):
            tempranks.append(ranks[i - 1])
            q = Q(tempranks, G_list_target, 'target')
            if q >= best:
                best = q
                sigma = i
        return rho, sigma, best, ranks


def process_node(node, G, lu, ablation_mode=False):
    ranks = ranking(node, G, lu)
    rho, sigma, q, ranks = Discovery(ranks, G, lu, ablation_mode=ablation_mode)
    return (node, rho, sigma, q, ranks)


def findGroups(G, k, lu, ablation_mode=False):
    out_rows = []

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(process_node, node, G, lu, ablation_mode): node
            for node in G.nodes
        }

        for future in tqdm.tqdm(as_completed(futures), total=len(G.nodes)):
            node_result = future.result()
            out_rows.append(node_result)

    out = pd.DataFrame(out_rows, columns=['node', 'rho', 'sigma', 'q', 'ranks']).set_index('node')

    out = out.sort_values(by=['q'], ascending=False)

    out['reference'] = [[] for _ in range(len(out))]
    out['subgroup'] = [[] for _ in range(len(out))]

    for index, row in out.iterrows():
        out.at[index, 'reference'] = [x[0] for x in row['ranks'][0:row['rho']]]
        out.at[index, 'subgroup'] = [x[0] for x in row['ranks'][0:row['sigma']]]

    return topK(out, k)

if __name__ == '__main__':
    with open('graph_349519.pkl', 'rb') as input:
        graph = pkl.load(input)

    # Extracting data
    edge_index = graph['edge_index']
    num_nodes = graph['num_nodes']

    # Create a networkx graph
    G = nx.Graph()

    # Add nodes
    G.add_nodes_from(range(num_nodes))

    # Add edges
    edges = list(zip(edge_index[0], edge_index[1]))
    G.add_edges_from(edges)


    # Create a lookup table for gower distances
    attributes = graph['node_feat']
    lu = pd.DataFrame(attributes)
    lu['target'] = lu[0] >= 6


    result = findGroups(G, 20, lu)
    result.to_csv('no_topk.csv')