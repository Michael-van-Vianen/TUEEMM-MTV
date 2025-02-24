import numpy as np
from numpy import floating
import pandas as pd

from typing import Any, List, Tuple, Dict


def compare_subgroups(df_normal: pd.DataFrame, df_ablation: pd.DataFrame, k: int = 10) -> Dict[str, float]:
    """
    Compares subgroups between normal and ablation runs.

    Args:
        df_normal (pd.DataFrame): DataFrame of normal run results.
        df_ablation (pd.DataFrame): DataFrame of ablation run results.
        k (int): Number of top rows to compare.

    Returns:
        Dict[str, float]: Summary metrics such as 'avg_q_normal' and 'avg_subgroup_overlap'.
    """
    top_normal = df_normal.head(k).reset_index(drop=True)
    top_ablation = df_ablation.head(k).reset_index(drop=True)

    avg_q_normal = top_normal['q'].mean()
    avg_q_ablation = top_ablation['q'].mean()

    avg_ref_size_normal = top_normal['reference'].apply(len).mean()
    avg_ref_size_ablation = top_ablation['reference'].apply(len).mean()

    avg_sub_size_normal = top_normal['subgroup'].apply(len).mean()
    avg_sub_size_ablation = top_ablation['subgroup'].apply(len).mean()

    overlap_scores = []
    for i in range(k):
        normal_sub = set(top_normal.loc[i, 'subgroup'])
        ablation_sub = set(top_ablation.loc[i, 'subgroup'])
        if len(normal_sub.union(ablation_sub)) > 0:
            overlap = len(normal_sub.intersection(ablation_sub)) / len(normal_sub.union(ablation_sub))
        else:
            overlap = 0.0
        overlap_scores.append(overlap)
    avg_subgroup_overlap = np.mean(overlap_scores)

    summary = {
        'avg_q_normal': avg_q_normal,
        'avg_q_ablation': avg_q_ablation,
        'avg_ref_size_normal': avg_ref_size_normal,
        'avg_ref_size_ablation': avg_ref_size_ablation,
        'avg_sub_size_normal': avg_sub_size_normal,
        'avg_sub_size_ablation': avg_sub_size_ablation,
        'avg_subgroup_overlap': avg_subgroup_overlap
    }
    return summary


def evaluate_pattern_subgroups(df: pd.DataFrame, lu: pd.DataFrame, k: int = 10) -> tuple[floating[Any], list[float]]:
    """
    Evaluates pattern subgroups in a DataFrame.

    Args:
        df (pd.DataFrame): Ranked results, containing subgroup data.
        lu (pd.DataFrame): Lookup with 'target' column.
        k (int): Number of top results to process.

    Returns:
        Tuple[float, List[float]]: Average target ratio as a float and a list of ratios.
    """
    top_k = df.head(k)

    target_ratios = []
    for idx, row in top_k.iterrows():
        subgroup_nodes = row['subgroup']
        n_target_true = sum(lu.loc[node, 'target'] for node in subgroup_nodes)
        ratio_true = n_target_true / max(1, len(subgroup_nodes))
        target_ratios.append(ratio_true)

    return np.mean(target_ratios), target_ratios