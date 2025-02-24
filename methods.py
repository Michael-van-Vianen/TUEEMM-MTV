import copy
import pandas as pd
import numpy as np
import tqdm
import networkx as nx
import pickle as pkl
import gower

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, List, Tuple


def top_k_func(groups: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    Selects the top k rows from the dataset while removing rows
    that have high overlap in 'reference' and 'subgroup' columns.

    Args:
        groups (pd.DataFrame): The DataFrame containing group information.
        k (int): The number of rows to keep after filtering.

    Returns:
        pd.DataFrame: The filtered top k rows.
    """
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


def get_attributes(x: int, lu: pd.DataFrame) -> List[float]:
    """
    Retrieves numeric attribute values from a lookup DataFrame for a single node.

    Args:
        x (int): The node identifier.
        lu (pd.DataFrame): The lookup DataFrame with numeric columns.

    Returns:
        List[float]: A list of numeric values for the node.
    """
    numeric_cols = lu.select_dtypes(include=[np.number]).columns
    return [float(i) for i in lu.loc[x, numeric_cols].tolist()]


def ranking(x: int, s: nx.Graph, lu: pd.DataFrame) -> List[Tuple[Any, Any]]:
    """
    Computes a ranking of other nodes based on shortest path lengths
    and breaks ties using gower distances.

    Args:
        x (int): The reference node.
        s (nx.Graph): The network graph.
        lu (pd.DataFrame): The lookup DataFrame with attributes.

    Returns:
        List[Tuple[Any, Any]]: A list of tuples (node, target) in ranked order.
    """
    d_dict = nx.shortest_path_length(s, x)
    distances = list(d_dict.values())
    D = list(zip(d_dict.keys(), d_dict.values()))

    for d in np.unique(distances):
        left = distances.index(d)
        right = len(distances) - distances[::-1].index(d)
        tie_list = D[left:right]

        gower_list = gower.gower_matrix(
            pd.DataFrame([get_attributes(x, lu)]),
            pd.DataFrame([get_attributes(t[0], lu) for t in tie_list])
        )
        sorted_gower = [node for _, node in sorted(zip(gower_list[0], tie_list))]
        D[left:right] = sorted_gower

    ranks = [(node[0], lu.loc[node[0]]['target']) for node in D]
    return ranks


def q_func(s: List[Tuple[Any, int]], g: List[Tuple[Any, int]], target: str) -> float:
    """
    Computes the WRAcc-based quality measure Q for a given subgroup.

    Args:
        s (List[Tuple[Any, int]]): The subgroup's (node, target) pairs.
        g (List[Tuple[Any, int]]): The whole population's (node, target) pairs.
        target (str): Not used in this function, included for consistency.

    Returns:
        float: The quality measure Q.
    """
    s_size = len(s)
    g_size = len(g)
    if s_size == 0 or g_size == 0:
        return 0

    cover = s_size / g_size
    n_target_s = sum([x[1] for x in s])
    n_target_g = sum([x[1] for x in g])

    wr_acc = (cover ** 0.5) * ((n_target_s / s_size) - (n_target_g / g_size))
    return abs(wr_acc)


def discovery(ranks: List[Tuple[Any, Any]],
              g: nx.Graph,
              lu: pd.DataFrame,
              ablation_mode: bool = False
              ) -> Tuple[int, int, float, List[Tuple[Any, Any]]]:
    """
    Determines thresholds rho and sigma based on the Q measure.

    Args:
        ranks (List[Tuple[Any, Any]]): Ranked list of (node, target).
        g (nx.Graph): The whole graph.
        lu (pd.DataFrame): Lookup DataFrame.
        ablation_mode (bool): If True, adjusts the search for sigma.

    Returns:
        Tuple[int, int, float, List[Tuple[Any, Any]]]:
            (rho, sigma, best_quality, ranks)
    """
    g_list_target = list(zip(list(g.nodes), [lu.loc[x]['target'] for x in list(g.nodes)]))

    if not ablation_mode:
        rho = 0
        sigma = 0
        best = 0

        temp_ranks = [x for x in ranks[0:4]]
        for i in range(5, len(ranks) + 1):
            temp_ranks.append(ranks[i - 1])
            q = q_func(temp_ranks, g_list_target, 'target')
            if q >= best:
                best = q
                rho = i

        best = 0
        temp_ranks = [x for x in ranks[0:2]]
        for i in range(3, rho + 1):
            temp_ranks.append(ranks[i - 1])
            q = q_func(temp_ranks, [x for x in ranks[0:rho]], 'target')
            if q >= best:
                best = q
                sigma = i

        return rho, sigma, best, ranks

    else:
        rho = len(ranks)
        best = 0
        sigma = 0
        temp_ranks = [x for x in ranks[0:4]]
        for i in range(5, rho + 1):
            temp_ranks.append(ranks[i - 1])
            q = q_func(temp_ranks, g_list_target, 'target')
            if q >= best:
                best = q
                sigma = i
        return rho, sigma, best, ranks


def process_node(node: int,
                 g: nx.Graph,
                 lu: pd.DataFrame,
                 ablation_mode: bool = False
                 ) -> Tuple[int, int, int, float, List[Tuple[Any, Any]]]:
    """
    Processes a single node by computing its ranking and subgroup qualities.

    Args:
        node (int): The node to process.
        g (nx.Graph): The whole graph.
        lu (pd.DataFrame): Lookup DataFrame with attributes.
        ablation_mode (bool): If True, adjusts discovery logic.

    Returns:
        Tuple[int, int, int, float, List[Tuple[Any, Any]]]:
            (node, rho, sigma, q, ranks)
    """
    ranks = ranking(node, g, lu)
    rho, sigma, q, ranks = discovery(ranks, g, lu, ablation_mode=ablation_mode)
    return node, rho, sigma, q, ranks


def find_groups(g: nx.Graph,
                k: int,
                lu: pd.DataFrame,
                ablation_mode: bool = False,
                use_multiprocessing: bool = True
                ) -> pd.DataFrame:
    """
    Finds subgroups in the graph, either in parallel (multiprocessing) or 
    single-threaded, based on the 'use_multiprocessing' parameter.

    Args:
        g (nx.Graph): The whole graph to analyze.
        k (int): Number of top groups to return.
        lu (pd.DataFrame): Lookup DataFrame with attributes.
        ablation_mode (bool): If True, uses ablation variant of discovery.
        use_multiprocessing (bool): Whether to enable multiprocessing.

    Returns:
        pd.DataFrame: Filtered top k rows with references and subgroups.
    """
    out_rows = []

    if use_multiprocessing:
        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(process_node, node, g, lu, ablation_mode): node
                for node in g.nodes
            }

            for future in tqdm.tqdm(as_completed(futures), total=len(g.nodes)):
                node_result = future.result()
                out_rows.append(node_result)
    else:
        for node in tqdm.tqdm(g.nodes):
            node_result = process_node(node, g, lu, ablation_mode)
            out_rows.append(node_result)

    out = pd.DataFrame(out_rows, columns=['node', 'rho', 'sigma', 'q', 'ranks']).set_index('node')

    out = out.sort_values(by=['q'], ascending=False)

    out['reference'] = [[] for _ in range(len(out))]
    out['subgroup'] = [[] for _ in range(len(out))]

    for index, row in out.iterrows():
        out.at[index, 'reference'] = [x[0] for x in row['ranks'][0:row['rho']]]
        out.at[index, 'subgroup'] = [x[0] for x in row['ranks'][0:row['sigma']]]

    return top_k_func(out, k)


if __name__ == '__main__':
    with open('graph_349519.pkl', 'rb') as input_file:
        graph = pkl.load(input_file)

    edge_index = graph['edge_index']
    num_nodes = graph['num_nodes']

    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edges = list(zip(edge_index[0], edge_index[1]))
    G.add_edges_from(edges)

    attributes = graph['node_feat']
    lookup_df = pd.DataFrame(attributes)
    lookup_df['target'] = lookup_df[0] >= 6

    USE_MULTIPROCESSING = False

    result = find_groups(G, 20, lookup_df, ablation_mode=False, use_multiprocessing=USE_MULTIPROCESSING)
    result.to_csv('no_topk.csv')
