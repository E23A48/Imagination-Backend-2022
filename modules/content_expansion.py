from serpapi import GoogleSearch
from modules.knowledge_resources import *
from modules.semantic_match import *
import requests
import json 

def contentBasedImageSearch(imgURL):
    params = {
    "engine": "google_reverse_image",
    "image_url": imgURL,
    "api_key": "a37557c4f0ab4a342ee7acaf220f6a301826dcd3d1a0f638650c757efbe874c9"
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    if "inline_images_serpapi_link" in results:
      inline_images = results['inline_images_serpapi_link']
    else:
       inline_images = ''
    return inline_images

def getImages(url, T_var):
  if(url != ''):
    res = requests.get(url,params = { "api_key": "52aa9ff40ec6de099df134e94f900778c0e32755f650eccd430ba56056a4c6ad"})
    response = json.loads(res.text)
    response = response['images_results'][:T_var]
  else:
    response = []
  return response

def makeArray(best_images, response_images):
  result_array = []
  for i in range(len(best_images)):
    temp = []
    temp.append(best_images[i]['originalURL'])
    for img in response_images[i]:
      temp.append(img['thumbnail'])

    result_array.append(temp)

  return result_array

def contentExpansion(processed_sentences, best_images,T_var, D_var,M_var):
  print('CONTENT EXPANSION')
  response_images=[]
  for obj in best_images:
    image_being_compared = obj
    
    searched_images_URL = contentBasedImageSearch(image_being_compared['originalURL'])

    response_images.append(getImages(searched_images_URL, T_var))


  # print("RESPONSE IMAGES",len(response_images))
  result_array = makeArray(best_images,response_images)
  # print('RESULT ARRAY',len(result_array))

  results_captions =getCandidateImagesContentBased(result_array, T_var, D_var)
  # print('RESULTS_CAPTIONS', len(results_captions))
  best_images_final = sentenceSimilarity(processed_sentences, results_captions,M_var)
  
  return best_images_final
   
