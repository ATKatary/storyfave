import math
from pathlib import Path
from matplotlib import pyplot as plt 
from transformers import CLIPTokenizer

BASE_DIR = Path(__file__).resolve().parent
SDV1_5 = "runwayml/stable-diffusion-v1-5"
default_tokenizer = CLIPTokenizer.from_pretrained(SDV1_5, subfolder="tokenizer")

def display(imgs, r, c=-1):
    n = len(imgs)
    if c == -1: c = math.ceil(r / n)

    fig = plt.figure(figsize=(10, 7)) 
    fig.add_subplot(r, c, 1) 

    for i in range(r):
        for j in range(c):
            fig.add_subplot(r, c, i*r + j + 1) 
            
            plt.imshow(imgs[i*r + j]['img']) 
            plt.axis('off') 
            plt.title(imgs[i*r + j]['label'])

