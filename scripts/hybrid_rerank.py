"""Hybrid BM25 + TF-IDF + dense retrieval with reranking (final method)."""
import math
import numpy as np
from collections import Counter
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from data_utils import load_data, tokenize, expand_query
from metric import ndcg_at_5

BASE = Path(__file__).parent.parent / 'data' / 'raw'


class BM25:
    """Okapi BM25 ranking function."""

    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.N = len(corpus)
        self.avgdl = sum(len(d) for d in corpus) / max(1, self.N)
        self.doc_len = [len(d) for d in corpus]
        self.doc_freqs = [Counter(d) for d in corpus]

        df = Counter()
        for d in corpus:
            for w in set(d):
                df[w] += 1
        self.idf = {
            w: math.log(1 + (self.N - n + 0.5) / (n + 0.5))
            for w, n in df.items()
        }

    def get_scores(self, q_tokens):
        s = np.zeros(self.N)
        for w in q_tokens:
            idf = self.idf.get(w)
            if idf is None:
                continue
            for i, freq in enumerate(self.doc_freqs):
                f = freq.get(w, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                s[i] += idf * (f * (self.k1 + 1)) / denom
        return s


def minmax(m):
    """Row-wise min-max normalisation."""
    mn = m.min(axis=1, keepdims=True)
    mx = m.max(axis=1, keepdims=True)
    return (m - mn) / (mx - mn + 1e-9)


def main():
    docs, train_q, qrels, test_q = load_data(BASE)

    # --- Prepare document text ---
    docs['title'] = docs['title'].fillna('').astype(str)
    docs['text']  = docs['text'].fillna('').astype(str)
    docs['boosted'] = (docs['title'] + '. ') * 3 + docs['text']
    doc_ids = docs['document_id'].to_numpy()

    # --- BM25 index ---
    bm25 = BM25([tokenize(t) for t in docs['boosted']])

    # --- TF-IDF index ---
    tfidf_vec = TfidfVectorizer(
        stop_words='english', ngram_range=(1, 2),
        min_df=2, sublinear_tf=True,
    )
    tfidf_doc = normalize(tfidf_vec.fit_transform(docs['boosted']))

    # --- Dense embeddings ---
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    doc_emb = model.encode(
        docs['boosted'].tolist(),
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=True,
    )

    # --- Retrieval function ---
    def retrieve(query_frame):
        queries = [expand_query(q) for q in query_frame['query'].fillna('')]

        bm25_scores = np.vstack([bm25.get_scores(tokenize(q)) for q in queries])
        tfidf_scores = (normalize(tfidf_vec.transform(queries)) @ tfidf_doc.T).toarray()
        dense_scores = model.encode(queries, normalize_embeddings=True) @ doc_emb.T

        combined = (0.25 * minmax(bm25_scores)
                    + 0.25 * minmax(tfidf_scores)
                    + 0.50 * minmax(dense_scores))

        results = {}
        for i, qid in enumerate(query_frame['query_id']):
            top = np.argpartition(-combined[i], 4)[:5]
            top = top[np.argsort(-combined[i][top])]
            results[qid] = doc_ids[top].tolist()
        return results

    # --- Evaluate on training queries ---
    train_preds = retrieve(train_q)
    score, _ = ndcg_at_5(train_preds, qrels)
    print(f'Hybrid training nDCG@5 = {score:.4f}')


if __name__ == '__main__':
    main()