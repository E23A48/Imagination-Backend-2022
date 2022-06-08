import enum
from typing import List
from starlette.responses import StreamingResponse
from fastapi import FastAPI
from pydantic import BaseModel
from modules.preprocessing import preprocessStory
from modules.knowledge_extraction import knowledgeExtractor
from modules.content_expansion import contentExpansion
from modules.query_expansion import queryExpansion
from modules.semantic_match import sentenceSimilarity
from modules.knowledge_resources import getCandidateImages
from modules.knowledge_resources import getCandidateImagesContentBased
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
# from pdfCreator import createPDF
from starlette.responses import FileResponse
import requests

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Payload(BaseModel):
    story: str
    

@app.get("/")
async def root():
    return {"message": "Hello World"}

def downloadImages(url_images):
    for idx,image_url in enumerate(url_images):
        img_data = requests.get(image_url).content
        with open(f'image_name_{idx}.jpg', 'wb') as handler:
            handler.write(img_data)

@app.post("/imagination")
async def root(payload: Payload):

    K_var = 0
    N_var = 2
    T_var = 0
    D_var = 'COCO'
    M_var = 'COSINE'

    import timeit

    start = timeit.default_timer()

    # Módulo 1: Preprocesamiento del Texto
    response_en, response_es, processed_sentences = preprocessStory(payload.story)

    # Módulo 2: Extracción del Conocimiento
    response_tokens = knowledgeExtractor(processed_sentences)

   # Módulo 3: Generación de Query
    response_queries = queryExpansion(response_tokens, K_var)

    images = getCandidateImages(response_queries, N_var, D_var)

    best_images = sentenceSimilarity(processed_sentences, images,M_var)

    winner_images = contentExpansion(processed_sentences, best_images,T_var, D_var,M_var)

    return {"result": {
        "original_story": [x.capitalize() for x in response_es], 
        "english_story": [x.capitalize() for x in response_en], 
        "proccessed_sentences": processed_sentences,
        "tokens": response_tokens,
        "images": images,
        "winner_images": winner_images,
        "best_images": best_images,
        "url_images": [x["imageUrl"] for x in best_images]
    }}

class Payload2(BaseModel):
    sentences:list = []
    images:list = []



@app.post("/createPDF")
async def root(payload: Payload2):
    pdfName = createPDF(payload.images,payload.sentences)
    return FileResponse(pdfName, media_type='application/octet-stream',filename="file_name.pdf")