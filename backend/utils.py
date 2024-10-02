from pathlib import Path
from transformers import CLIPTokenizer

BASE_DIR = Path(__file__).resolve().parent
SDV1_5 = "runwayml/stable-diffusion-v1-5"
default_tokenizer = CLIPTokenizer.from_pretrained(SDV1_5, subfolder="tokenizer")