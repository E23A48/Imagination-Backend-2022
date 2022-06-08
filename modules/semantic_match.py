import math
import re
from collections import Counter
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
import nltk

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('wordnet')


# Esta función permite mapear los tokens obtenidos
# a definiciones de tokens de wordnet
def penn_to_wn(tag):
    if tag.startswith('N'):
        return wn.NOUN
 
    if tag.startswith('V'):
        return wn.VERB
 
    if tag.startswith('J'):
        return wn.ADJ
 
    if tag.startswith('R'):
        return wn.ADV
 
    return None
 
# Esta función extrae los synsets de una palabra determinado
# dado su token
def tagged_to_synset(word, tag):
    wn_tag = penn_to_wn(tag)
    if wn_tag is None:
        return None

    try:
        return wn.synsets(word, wn_tag)[0]
    except:
        return None
 
def sentence_similarityWORDNET(sentence1, sentence2):
    # Tokenización y etiquetado
    sentence1 = pos_tag(word_tokenize(sentence1))
    sentence2 = pos_tag(word_tokenize(sentence2))

    # Obtener los synsets
    synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]
    synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]

    # Filtro de synsets
    synsets1 = [ss for ss in synsets1 if ss]
    synsets2 = [ss for ss in synsets2 if ss]

    score, count = 0.0, 0
 
    # Para cada palabra en los synsets
    for synset in synsets1:
        simlist = [synset.path_similarity(ss) for ss in synsets2 if synset.path_similarity(ss) is not None]
        if not simlist:
            continue;
        best_score = max(simlist)
 
        # Verificar si se calculo la similitud
        if best_score is not None:
            score += best_score
            count += 1
 
    # Media de los valores
    score /= count
    return score
 



def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    # print(intersection)
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

WORD = re.compile(r"\w+")
def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)


def sentenceSimilarity(original_sentences, captions,M_var):
    results = []

    for idx in range(len(original_sentences)):
    # for idx in range(1):

        sentence = original_sentences[idx]
        vector_sentence = text_to_vector(sentence.lower())
        current_image_captions = captions[idx]
        temp = {
            'sentence':'',
            'rating':-1
        }

        # print('SENTENCE', idx)

        for curr in current_image_captions:
            sen = text_to_vector(curr['caption'].lower())

            # print(sentence , get_cosine(vector_sentence, sen), curr['caption'].lower(), curr['originalURL'])
            score = -1
            if(M_var == 'COSINE'):
              score = get_cosine(vector_sentence, sen)
            else:
              score = sentence_similarityWORDNET(sentence ,curr['caption'].lower())

            if  score> temp['rating']:
                temp = {
                    'caption':curr['caption'].lower(),
                    'sentence':sentence,
                    'imageUrl':curr['imageUrl'],
                    'originalURL':curr['originalURL'],
                    'rating':round(score,3)}
        
        results.append(temp)

    # print(results)

    return results