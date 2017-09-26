import numpy as np
from nltk.tokenize import TreebankWordTokenizer as Tok
from collections import defaultdict


def load_embs(filename):
    embs = {}
    with open(filename) as f:
        for line in f:
            toks = line.split()
            embs[toks[0]] = np.array([float(t) for t in toks[1:]])
    return embs


def load_embs_matrix(filename):
    data = np.loadtxt(filename, delimiter=' ', comments=None, dtype=object)
    words = data[:, 0]
    embs = np.array(data[:, 1:], dtype=float)
    embs = np.concatenate(([[0.0] * embs.shape[1]], embs))
    words = defaultdict(int, [(word, i+1) for i, word in enumerate(words)])
    return embs, words


def get_indices(sent, words):
    indices = []
    for word in sent:
        try:
            indices.append(words[word])
        except KeyError: #unk
            indices.append(0)
    return indices


def preprocess_sent(sent):
    """
    Take a sentence in string format and returns
    a list of lemmatized tokens.
    """
    #tokenized = word_tokenize(sent.lower())
    tokenizer = Tok()
    tokenized = tokenizer.tokenize(sent.lower())
    return tokenized


def average_sent(sent, embs):
    result = []
    for word in sent:
        try:
            result.append(embs[word])
        except KeyError:
            pass
    return np.mean(result, axis=0)
    
    
