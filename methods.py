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

def topK(groups, k):
    top_k = copy.deepcopy(groups)
    i = 0
    while i < len(top_k) and i < k:
        top_group = top_k.iloc[i]
        ref_group = set(top_group['reference'])
        sub_group = set(top_group['subgroup'])

        rows_to_drop = []
        for idx, row in top_k.iloc[i+1:].iterrows():
            overlap_ref = len(ref_group.intersection(row['reference'])) / len(ref_group)
            overlap_sub = len(sub_group.intersection(row['subgroup'])) / len(sub_group)
            if overlap_ref > 0.75 and overlap_sub > 0.75:
                rows_to_drop.append(idx)

        top_k = top_k.drop(rows_to_drop)
        i += 1

    return top_k.iloc[:k]


def findGroups(G, k, lu, ablation_mode=False):
    out = pd.DataFrame(columns=['rho', 'sigma', 'q', 'ranks'])

    for node in tqdm.tqdm(list(G.nodes)):
        rho, sigma, q, ranks = Discovery(ranking(node, G, lu), G, lu, ablation_mode=ablation_mode)
        out.loc[node] = [rho, sigma, q, ranks] 

    out = out.sort_values(by=['q'], ascending=False)
    out['reference'] = [[] for _ in range(len(G))]
    out['subgroup'] = [[] for _ in range(len(G))]

    for index, row in out.iterrows():
        out.at[index, 'reference'] = [x[0] for x in row['ranks'][0:row['rho']]]
        out.at[index, 'subgroup'] = [x[0] for x in row['ranks'][0:row['sigma']]]

    return topK(out, k)


def prototype(G):
    prot = rand.choice([n for n in G])
    return prot


def getAttributes(x, lu):
    numeric_cols = lu.select_dtypes(include=[np.number]).columns
    return [float(i) for i in lu.loc[x, numeric_cols].tolist()]


def ranking(x, S, lu):
    D = nx.shortest_path_length(S, x)
    distances = list(D.values())
    D = list(zip(D.keys(),D.values()))

    for d in np.unique(distances):
    
        left, right = distances.index(d), len(distances) - distances[::-1].index(d)
        tielist = D[left:right]
        gowerlist = gower.gower_matrix(pd.DataFrame([getAttributes(x, lu)]), pd.DataFrame([getAttributes(i[0], lu) for i in tielist]))
        sorted_gower = [x for _,x in sorted(zip(gowerlist[0],tielist))]
        D[left:right] = sorted_gower
    
    ranks = [(x[0], lu.loc[x[0]]['target']) for x in D]

    return ranks

def Q(S, G, target):
    S_size = len(S)
    G_size = len(G)
    cover = S_size / G_size
    
    n_target_S = sum([x[1] for x in S])
    n_target_G = sum([x[1] for x in G])
    
    WRAcc = (cover**0.5)* ((n_target_S/S_size) - (n_target_G/G_size))
    RAcc = ((n_target_S/S_size) - (n_target_G/G_size))

    return abs(WRAcc)

def Q2(S, G, target):
    cover = len(S) / len(G)

    target_counts_S = sum([x[1] for x in S])
    target_counts_G = sum([x[1] for x in G])
    ratio_s = target_counts_S/len(S)
    ratio_g = target_counts_G/len(G)

    #Reindex
    target_counts_S = np.array([ratio_s, 1-ratio_s])
    target_counts_G = np.array([ratio_g, 1-ratio_g])
    
    non_zero_mask = (target_counts_S > 0)

    KL_divergence = np.sum(target_counts_S[non_zero_mask] * np.log(target_counts_S[non_zero_mask] / target_counts_G[non_zero_mask]))
    
    WKL = cover * KL_divergence
    
    return WKL

def QTest(S, G, target):
    return rand.uniform(0,1)


def Discovery(ranks, G, lu, ablation_mode=False):
    
    G_list_target = list(zip(list(G.nodes), [lu.loc[x]['target'] for x in list(G.nodes)]))

    
    if not ablation_mode:
        rho = 0
        sigma = 0
        best = 0

        tempranks = [x for x in ranks[0:4]]

        for i in range(5, len(ranks)):
            tempranks.append(ranks[i - 1])
            q = Q(tempranks, G_list_target, 'target')
            if q >= best:
                best = q
                rho = i

        best = 0
        tempranks = [x for x in ranks[0:2]]
        for i in range(3, rho):
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
        for i in range(5, rho):
            tempranks.append(ranks[i - 1])
            q = Q(tempranks, G_list_target, 'target')
            if q >= best:
                best = q
                sigma = i

        return rho, sigma, best, ranks


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