import nltk
from nltk.corpus import wordnet
import random
import itertools
nltk.download('wordnet')

try:
    nltk.word_tokenize('foobar')
except LookupError:
    nltk.download('punkt')
try:
    nltk.pos_tag(nltk.word_tokenize('foobar'))
except LookupError:
    nltk.download('averaged_perceptron_tagger')

def get_some_word_synonyms(word):
    word = word.lower()
    synonyms = []
    synsets = wordnet.synsets(word)
    if (len(synsets) == 0):
        return []
    synset = synsets[0]
    lemma_names = synset.lemma_names()
    for lemma_name in lemma_names:
        lemma_name = lemma_name.lower().replace('_', ' ')
        if (lemma_name != word and lemma_name not in synonyms):
            synonyms.append(lemma_name)
    return synonyms


def getSyns(tokens):
  syns = {x: [] for x in tokens}
  for token in tokens:
    syn = get_some_word_synonyms(token)
    syns[token].append(token)
    for s in syn:
      syns[token].append(s)
  return syns

def queryExpansion(tokens, n = 5):

    rended_queries = []

    for tks in tokens:
        query_syns = getSyns(tks)
        query_syns_s = [query_syns[x] for x in query_syns]
        query_sysns_f = list(itertools.product(*query_syns_s))
        random.shuffle(query_sysns_f)
        new_queries = query_sysns_f[:n]
        #new_queries = []
        new_queries.append(tuple(tks))

        rended_queries.append([', '.join(x) for x in new_queries])

    return rended_queries