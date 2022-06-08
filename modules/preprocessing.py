import re
import unidecode 
from unicodedata import normalize
from googletrans import Translator
# import nltk
# from nltk.corpus import stopwords
# nltk.download('stopwords')

def translateStory(story):
  translator = Translator()
  return translator.translate(story).text

def separateSentences(story):
  return re.split(r'(?<=\.) ', story)

def preprocessStory(story):
  # stpw = stopwords.words('english')
  translatedStory = translateStory(story)
  separated_sentences = separateSentences(translatedStory)
  processed_sentences = []
  for sentence in separated_sentences:
    #eliminando masyusculas
    X = sentence.lower()
    #sacar acentos
    X = unidecode.unidecode(X)
    #Sin caracteres especiales
    X= re.sub(u'[^a-zA-ZáéíóúÁÉÍÓÚâêîôÂÊÎÔãõÃÕçÇñ@# ]', '', X) 
    #Arreglo espacios innecesarios
    X = re.sub(' +', ' ', X)
    # #sacar stopwords
    # X= " ".join([word for word in X.split() if word.lower() not in stpw])

    processed_sentences.append(X)

  return separated_sentences, separateSentences(story), processed_sentences