import requests
import shutil
from models.oscar import inference 
from serpapi import GoogleSearch
import uuid
import clip
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from models.oscar import ClipCaptionModel, generate2
from huggingface_hub import hf_hub_download
import torch
import torch.nn.functional as nnf
import os
from torch import nn
import numpy as np
import sys
from typing import Tuple, List, Union, Optional
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import skimage.io as io
import PIL.Image

N = type(None)
V = np.array
ARRAY = np.ndarray
ARRAYS = Union[Tuple[ARRAY, ...], List[ARRAY]]
VS = Union[Tuple[V, ...], List[V]]
VN = Union[V, N]
VNS = Union[VS, N]
T = torch.Tensor
TS = Union[Tuple[T, ...], List[T]]
TN = Optional[T]
TNS = Union[Tuple[TN, ...], List[TN]]
TSN = Optional[TS]
TA = Union[T, ARRAY]


D = torch.device
CPU = torch.device('cpu')

def downloadImage(image_url):

  filename = str(uuid.uuid4())+'.jfif'

  r = requests.get(image_url, stream = True)

  if r.status_code == 200:
      r.raw.decode_content = True
      
      with open('./static/'+filename,'wb') as f:
          shutil.copyfileobj(r.raw, f)
          
      return './static/'+filename
  else:
      return None

def imageRetrievalBing(keywords):
    subscription_key = "ce151b40503a4664b2690d93920235ca"
    endpoint_url = "https://api.bing.microsoft.com/v7.0/images/search"
    headers = {"Ocp-Apim-Subscription-Key" : subscription_key}
    url_images = []
    for x in keywords:
        params  = {"q": x}
        response = requests.get(endpoint_url, headers=headers, params=params)
        response.raise_for_status()

        search_results = response.json()
        url_images.append(search_results["value"][0]["thumbnailUrl"])

    return url_images

def bingSearchImagesByQuery(queries, n_top):
    subscription_key = "ce151b40503a4664b2690d93920235ca"
    endpoint_url = "https://api.bing.microsoft.com/v7.0/images/search"
    headers = {"Ocp-Apim-Subscription-Key" : subscription_key}

    url_images = []
    params  = {"q": queries}
    response = requests.get(endpoint_url, headers=headers, params=params)
    response.raise_for_status()

    search_results = response.json()
    url_images = [q["thumbnailUrl"] for q in search_results["value"][:n_top]]

    return url_images

def searchImagesByQuery(queries, n_top):
  params = {
    "engine": "google",
    "tbm": "isch",
    "ijn": 0,
    "q": queries,
    "api_key": "adc04dd5b8dd12c2fb86d4d52f7201827b32329039a9d1045076d70df455eac0"
  }
  search = GoogleSearch(params)
  results = search.get_dict()
  organic_results = results['images_results']

  thumbnails_result_top = [q['thumbnail'] for q in organic_results[:n_top]]

  return thumbnails_result_top
  


def getCandidateImages(queries, n = 4, model_name="COCO"):

  captions = [[] for x in range(len(queries))]

  is_gpu = False
  device = CUDA(0) if is_gpu else "cpu"
  clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
  tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
  prefix_length = 10
    
  model = ClipCaptionModel(prefix_length)

  coco_weight = hf_hub_download(repo_id="akhaliq/CLIP-prefix-captioning-COCO-weights", filename="coco_weights.pt")

  if model_name == "COCO":
    model_path = coco_weight
  else:
    model_path = conceptual_weight

  model.load_state_dict(torch.load(model_path, map_location=CPU))
  model = model.eval() 
  device = CUDA(0) if is_gpu else "cpu"
  model = model.to(device)

  use_beam_search = False

  for i in range(len(queries)):
    for j in range(len(queries[i])):

  # for i in range(1):
  #   for j in range(len(queries[i])):
      # Consulta de Imágenes por Query
      imgs = bingSearchImagesByQuery(queries[i][j], n)

      for img in imgs:
        # Descargar Imagen
        img_name = downloadImage(img)
        image = io.imread(img_name)
        pil_image = PIL.Image.fromarray(image)
        image = preprocess(pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
            prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)
        if use_beam_search:
            generated_text_prefix = generate_beam(model, tokenizer, embed=prefix_embed)[0]
        else:
            generated_text_prefix = generate2(model, tokenizer, embed=prefix_embed)
        
        # Caption Inference
        captions[i].append({
          'imageUrl': img_name,
          'originalURL':img,
          'caption': generated_text_prefix
        })
  
  return captions

def getCandidateImagesContentBased(imgs, n = 4, model_name="COCO"):
  print(len(imgs))
  captions = [[] for x in range(len(imgs))]

  is_gpu = False
  device = CUDA(0) if is_gpu else "cpu"
  clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
  tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
  prefix_length = 10
    
  model = ClipCaptionModel(prefix_length)

  coco_weight = hf_hub_download(repo_id="akhaliq/CLIP-prefix-captioning-COCO-weights", filename="coco_weights.pt")

  if model_name == "COCO":
    model_path = coco_weight
  else:
    model_path = conceptual_weight

  model.load_state_dict(torch.load(model_path, map_location=CPU))
  model = model.eval() 
  device = CUDA(0) if is_gpu else "cpu"
  model = model.to(device)

  use_beam_search = False

  # for i in range(len(queries)):
  #   for j in range(len(queries[i])):

  for i in range(len(imgs)):
    for img in imgs[i]:
      # Consulta de Imágenes por Query
      
      # Descargar Imagen
      print('THIS SHOULD BE URL', img)
      img_name = downloadImage(img)
      image = io.imread(img_name)
      pil_image = PIL.Image.fromarray(image)
      image = preprocess(pil_image).unsqueeze(0).to(device)
      with torch.no_grad():
          prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
          prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)
      if use_beam_search:
          generated_text_prefix = generate_beam(model, tokenizer, embed=prefix_embed)[0]
      else:
          generated_text_prefix = generate2(model, tokenizer, embed=prefix_embed)
      
      # Caption Inference
      captions[i].append({
        'imageUrl': img_name,
        'originalURL':img,
        'caption': generated_text_prefix
      })
  
  return captions