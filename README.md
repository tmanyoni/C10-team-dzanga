C10-dzanga — Zimbabwe Agricultural Extension RAG

Retrieval system for a RAG pipeline serving smallholder farmers in
Zimbabwe. Given a question about crops, pests, nutrients, soil, or
climate, the system ranks the five most relevant documents from an
agricultural knowledge base. Measured with nDCG@5.

Team: dzanga   Cohort: C10

Dataset

The corpus is documents.csv — 695 agricultural factsheets from FAO,
IITA, CGIAR, CABI, and synthetic CC0 rewrites. Each factsheet covers
one of six themes: crop diseases, pests, nutrient deficiencies, soil
management, climate adaptation, or fertiliser advice.

Development labels come from train_queries.csv (308 questions) and
qrels_train.csv (4,194 graded relevance labels 0-3), rated by
agricultural domain experts. Full ethics and bias analysis is in
docs/data_card.pdf.

Test queries are in test_queries.csv (200 questions, no labels).

Training Pipeline

Preprocessing. Titles are repeated 3 times and concatenated with body
text so short, high-signal titles get more weight. TF-IDF uses
lowercasing, stopword removal, unigrams and bigrams, min_df=2. BM25
uses standard tokenisation. A small synonym map handles agricultural
synonyms (maize and corn, flooding and waterlogging).

Methods tried.
1. TF-IDF baseline.
2. BM25 with grid search over k1 and b.
3. Dense retrieval with all-MiniLM-L6-v2.
4. Final: hybrid BM25 + TF-IDF + dense, then cross-encoder rerank.

Final design. Candidate retrieval returns the top-50 documents using
Reciprocal Rank Fusion of BM25, TF-IDF, and dense cosine scores. A
cross-encoder reranker reorders the top-50, and the top-5 are
submitted. Hyperparameters were tuned on training qrels with grouped
cross-validation so paraphrased queries stay in the same fold.

Evaluation

Metric. Local nDCG@5 matching the competition definition:
DCG@5 = sum of rel_i / log2(i+1), and nDCG@5 = DCG@5 / IDCG@5.

Protocol. All tuning used training qrels only. The final model was
scored on the hidden Kaggle test set. Submission format was validated
(1000 rows, 5 per query, no duplicates, valid IDs, query order
preserved).

Results. results/scores.md shows training nDCG@5 for each method.
The final hybrid + rerank pipeline outperformed the TF-IDF baseline
of 0.5131.

Reproduction

git clone https://github.com/tmanyoni/C10-team-dzanga.git
cd C10-dzanga
python -m venv .venv
source .venv/bin/activate    (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
python data/download_data.py
bash scripts/run_all.sh

This produces submission.csv, results/scores.md, and one JSON file
per method.

Appendix

Team members
- Taurai Manyoni - data pipeline, BM25, evaluation
- Perseverance Javangwe - embeddings, reranker, submission validation
- Taurai Manyoni - query expansion, error analysis, documentation
- Perseverance Javangwe - experiment tracking, reproducibility

Mentors
- Oluwasen Ajayi

Cohort Challenges (in docs/):
1. Problem Statement
2. Data Card
3. Impact Statement
4. Stakeholder Engagement Plan

Acknowledgements. Competition hosted by TRI AI. Factsheets are
CC-BY or synthetic CC0.