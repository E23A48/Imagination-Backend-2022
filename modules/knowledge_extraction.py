import nltk
import re
from nltk.corpus import wordnet as wn
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('omw-1.4')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

pos_tag_map = {
    'NN': [ wn.NOUN ],
    'JJ': [ wn.ADJ, wn.ADJ_SAT ],
    'RB': [ wn.ADV ],
    'VB': [ wn.VERB ]
}

def stopword_treatment(tokens):
  stopword = stopwords.words('english')
  result = []
  for token in tokens:
      if token[0].lower() not in stopword:
          result.append(tuple([token[0].lower(), token[1]]))
  return result

def underscore_replacer(tokens):
  new_tokens = {}
  for key in tokens.keys():
      mod_key = re.sub(r'_', ' ', key)
      new_tokens[mod_key] = tokens[key]
  return new_tokens

def tokenizer(sentence):
  return word_tokenize(sentence)

def pos_tagger(tokens):
  return nltk.pos_tag(tokens)

def knowledgeExtractor(proccessed_sentences):
  tks = [[] for x in range(len(proccessed_sentences))]

  for i in range(len(proccessed_sentences)):
    tokens = tokenizer(proccessed_sentences[i])
    tokens = pos_tagger(tokens)
    tokens = stopword_treatment(tokens)

    for tk in tokens:
      tks[i].append(tk[0])
  
  return tks