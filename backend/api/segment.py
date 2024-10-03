import cv2
import torch
import numpy as np
from PIL import Image
import supervision as sv
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
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
    

BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=2)
LABEL_ANNOTATOR = sv.LabelAnnotator(text_thickness=2, text_scale=1, text_color=sv.Color.BLACK)

def visualize_boxes(img, boxes, prompt, size=(3, 3), model="dino"):
    if model == "dino":
        xyxy = np.zeros(boxes['boxes'].shape)

        n = boxes['boxes'].shape[0]
        for i in range(n):
          for j in range(4):
            xyxy[i][j] = boxes['boxes'][i][j].item()

        class_map = {}
        class_id = np.zeros((n,), dtype=np.int64)
        for i in range(n):
          if boxes['labels'][i] not in class_map:
            class_map[boxes['labels'][i]] = len(class_map)
          class_id[i] = class_map[boxes['labels'][i]]
        
        detections = sv.Detections(
            xyxy=xyxy,
            class_id=class_id,
            confidence=boxes['scores'].cpu().detach().numpy()
        ).with_nms(threshold=0.1)

        labels = labels = [
            f"{class_id} {confidence:0.3f}"

            for class_id, confidence
            in zip(detections.class_id, detections.confidence)
        ]

    elif model == "yolo":
        prompt = prompt.split(" ")
        detections = sv.Detections.from_inference(boxes).with_nms(threshold=0.1)
        labels = [
            f"{prompt[class_id]} {confidence:0.3f}"

            for class_id, confidence
            in zip(detections.class_id, detections.confidence)
        ]
    else: raise ValueError("invalid model")

    annotated_image = img.copy()
    annotated_image = BOUNDING_BOX_ANNOTATOR.annotate(annotated_image, detections)
    annotated_image = LABEL_ANNOTATOR.annotate(annotated_image, detections, labels=labels)
    sv.plot_image(annotated_image, size)


    


