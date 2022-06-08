import os
from huggingface_hub import hf_hub_download
conceptual_weight = hf_hub_download(repo_id="akhaliq/CLIP-prefix-captioning-conceptual-weights", filename="conceptual_weights.pt")
coco_weight = hf_hub_download(repo_id="akhaliq/CLIP-prefix-captioning-COCO-weights", filename="coco_weights.pt")
