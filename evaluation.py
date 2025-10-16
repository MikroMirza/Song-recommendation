import numpy as np

def precision_at_k(recommended_ids, relevant_ids, k=10):
    recommended_ids = recommended_ids[:k]
    hits = len(set(recommended_ids) & set(relevant_ids))
    return hits / k

def recall_at_k(recommended_ids, relevant_ids, k=10):
    recommended_ids = recommended_ids[:k]
    hits = len(set(recommended_ids) & set(relevant_ids))
    return hits / len(relevant_ids) if len(relevant_ids) > 0 else 0

def ndcg_at_k(recommended_ids, relevant_ids, k=10):
    recommended_ids = recommended_ids[:k]
    dcg = 0
    for i, item in enumerate(recommended_ids):
        if item in relevant_ids:
            dcg += 1 / np.log2(i + 2)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant_ids), k)))
    return dcg / idcg if idcg > 0 else 0
