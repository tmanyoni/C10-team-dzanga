"""Baseline TF-IDF retrieval (Challenge-comparison reference)."""
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from data_utils import load_data
from metric import ndcg_at_5

# Data lives in ../data/raw relative to this script.
BASE = Path(__file__).parent.parent / 'data' / 'raw'


def main():
    docs, train_q, qrels, test_q = load_data(BASE)

    # Build text field: title + body.
    text = (docs['title'].fillna('') + '. ' + docs['text'].fillna(''))

    vec = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    doc_mat = normalize(vec.fit_transform(text))
    doc_ids = docs['document_id'].to_numpy()

    def retrieve(query_frame):
        q_mat = normalize(vec.transform(query_frame['query'].fillna('')))
        sims = q_mat @ doc_mat.T
        results = {}
        for i, qid in enumerate(query_frame['query_id']):
            row = sims.getrow(i).toarray().ravel()
            top = np.argpartition(-row, 4)[:5]
            top = top[np.argsort(-row[top])]
            results[qid] = doc_ids[top].tolist()
        return results

    train_preds = retrieve(train_q)
    score, _ = ndcg_at_5(train_preds, qrels)
    print(f'TF-IDF baseline training nDCG@5 = {score:.4f}')


if __name__ == '__main__':
    main()