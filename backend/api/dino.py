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

        with torch.no_grad(): prediction = model(**processed_img)

        return processor.post_process_grounded_object_detection(
            prediction,
            processed_img.input_ids,

            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=[img.size[::-1]]
        )
    elif model == "yolo":
        model = YOLOWorld(model_id="yolo_world/l")

        model.set_classes(prompt.split(" "))
        return model.infer(img)

    else: raise ValueError("invalid model")

