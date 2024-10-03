import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import supervision as sv
from inference.models.yolo_world.yolo_world import YOLOWorld

yolo = YOLOWorld(model_id="yolo_world/l")

model_id = "IDEA-Research/grounding-dino-tiny"
dino_processor = AutoProcessor.from_pretrained(model_id)
dino = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)

def get_bounding_boxes(img, prompt, model="dino", device="cuda"):
    if model == "dino":
        dino.to(device)
        processed_img = dino_processor(images=img, text=prompt, return_tensors="pt").to(device)

        with torch.no_grad(): prediction = dino(**processed_img)

        return dino_processor.post_process_grounded_object_detection(
            prediction,
            processed_img.input_ids,

            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=[img.size[::-1]]
        )
    elif model == "yolo":
        yolo.set_classes(prompt.split(" "))
        return yolo.infer(img)

    else: raise ValueError("invalid model")

