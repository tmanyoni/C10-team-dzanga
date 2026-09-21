"""Helper functions for loading data and cleaning text."""
import re
import pandas as pd
from pathlib import Path


def load_data(base):
    """Load the five competition CSVs from a folder."""
    base = Path(base)
    docs  = pd.read_csv(base / 'documents.csv')
    train = pd.read_csv(base / 'train_queries.csv')
    qrels = pd.read_csv(base / 'qrels_train.csv')
    test  = pd.read_csv(base / 'test_queries.csv')
    return docs, train, qrels, test


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(s):
    """Lowercase and split into word tokens."""
    return _TOKEN_RE.findall(str(s).lower())


# Simple synonym map for common agricultural terms.
SYNONYMS = {
    'maize':      ['corn'],
    'corn':       ['maize'],
    'groundnut':  ['peanut'],
    'peanut':     ['groundnut'],
    'fertiliser': ['fertilizer', 'nutrient'],
    'fertilizer': ['fertiliser', 'nutrient'],
    'flooding':   ['flood', 'waterlogging'],
    'flood':      ['flooding', 'waterlogging'],
    'drought':    ['dry', 'water stress'],
    'yellow':     ['chlorosis', 'yellowing'],
    'acidic':     ['acid', 'acidification'],
    'pest':       ['insect', 'borer'],
    'disease':    ['infection', 'blight', 'rust'],
}


def expand_query(q):
    """Add synonyms to a query to improve recall."""
    ql = str(q).lower()
    extras = [s for k, syns in SYNONYMS.items() if k in ql for s in syns]
    return str(q) + (' ' + ' '.join(extras) if extras else '')