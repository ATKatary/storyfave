import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset
from storyfave.backend.utils import *
from transformers import AutoProcessor
from transformers import AutoModelForCausalLM


checkpoint = "microsoft/git-base"
processor = AutoProcessor.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)

class CustomDataset(Dataset):
    def __init__(self, img_dir_path, prompt, tokenizer=default_tokenizer, verbose=False):
        self.tokenizer = tokenizer
        self.img_dir_path = img_dir_path 

        self.imgs, self.prompt = [], prompt
        for img in list(Path(img_dir_path).iterdir()):
            img = Image.open(img).convert("RGB")
            self.imgs.append(img)

            # pixel_values = processor(images=img, return_tensors="pt").to("cuda").pixel_values
            # generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
            # generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # self.prompts.append(prompt)
        self.n = len(self.imgs)
        self.img_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    
    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {
            "img": self.img_transforms(self.imgs[i % self.n]), 
            "prompt": self.tokenizer(
                self.prompt,
                truncation=True,
                padding="do_not_pad",
                max_length=self.tokenizer.model_max_length,
            ).input_ids
        }

def collate_fn(x):
    """ stacks images and prompts enabling us to load data in batches """
    img = torch.stack([x_i["img"] for x_i in x]).to(memory_format=torch.contiguous_format).float()
    
    input_ids = {"input_ids": [x_i["prompt"] for x_i in x]}
    prompt = default_tokenizer.pad(
        input_ids,
        return_tensors="pt",
        padding="max_length",
        max_length= default_tokenizer.model_max_length,
    ).input_ids

    return {
        "img": img,
        "prompt": prompt
    }


