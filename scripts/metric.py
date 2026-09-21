"""nDCG@5 metric for retrieval evaluation."""
import numpy as np


def dcg(relevances, k=5):
    """Discounted Cumulative Gain at cutoff k."""
    v = np.asarray(list(relevances)[:k], dtype=float)
    if len(v) == 0:
        return 0.0
    discounts = np.log2(np.arange(2, len(v) + 2))
    return float(np.sum(v / discounts))


def ndcg_at_5(predictions, qrels):
    """
    Compute mean nDCG@5 over all queries.

    predictions: dict {query_id: [ranked doc_ids]}
    qrels:       pandas DataFrame with columns query_id, document_id, relevance
    """
    lookup = {
        qid: dict(zip(group['document_id'], group['relevance']))
        for qid, group in qrels.groupby('query_id')
    }

    scores = {}
    for qid, judged in lookup.items():
        ranked = predictions.get(qid, [])[:5]
        pred_rel = [judged.get(d, 0) for d in ranked]
        ideal = sorted(judged.values(), reverse=True)[:5]
        idcg = dcg(ideal)
        scores[qid] = dcg(pred_rel) / idcg if idcg else 0.0

    return float(np.mean(list(scores.values()))), scores