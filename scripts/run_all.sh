#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
echo "=== Running baseline TF-IDF ==="
python baseline_tfidf.py
echo
echo "=== Running hybrid BM25 + dense retrieval ==="
python hybrid_rerank.py
echo
echo "Done."