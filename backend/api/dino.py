import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import supervision as sv
from inference.models.yolo_world.yolo_world import YOLOWorld

def get_bounding_boxes(img, prompt, model="dino", device="cuda"):
    if model == "dino":
        model_id = "IDEA-Research/grounding-dino-tiny"
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

        processed_img = processor(images=img, text=prompt, return_tensors="pt").to(device)

        with torch.no_grad(): 
            return model(**processed_img)
        
    elif model == "yolo":
        model = YOLOWorld(model_id="yolo_world/l")

        model.set_classes(prompt.split(" "))
        return model.infer(img)

    else: raise ValueError("invalid model")

