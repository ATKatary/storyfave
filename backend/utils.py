import math
from pathlib import Path
from matplotlib import pyplot as plt 
from transformers import CLIPTokenizer

BASE_DIR = Path(__file__).resolve().parent
SDV1_5 = "runwayml/stable-diffusion-v1-5"
default_tokenizer = CLIPTokenizer.from_pretrained(SDV1_5, subfolder="tokenizer")

def display(imgs, r, c=-1, w=3, h=3):
    n = len(imgs)
    if c == -1: c = math.ceil(r / n)

    fig = plt.figure(figsize=(w*c, h)) 
    fig.add_subplot(r, c, 1)
    fig.patch.set_visible(False)

    for i in range(n):
        fig.add_subplot(r, c, i + 1) 
        plt.imshow(imgs[i]['img']) 
        plt.axis("tight")
        plt.title(imgs[i]['label'])

