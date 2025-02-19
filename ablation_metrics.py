import numpy as np

def compare_subgroups(df_normal, df_ablation, k=10):
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
        # convert from list to set for overlap
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

def evaluate_pattern_subgroups(df, lu, k=10):
    top_k = df.head(k)

    target_ratios = []
    for idx, row in top_k.iterrows():
        subgroup_nodes = row['subgroup']
        n_target_true = sum(lu.loc[node, 'target'] for node in subgroup_nodes)
        ratio_true = n_target_true / max(1, len(subgroup_nodes))
        target_ratios.append(ratio_true)

    return np.mean(target_ratios), target_ratios