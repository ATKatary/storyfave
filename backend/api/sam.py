import torch
from PIL import Image
import requests
from transformers import SamModel, SamProcessor

def get_masks(img, boxes, device="cuda"):
    model_id = "facebook/sam-vit-huge"
    processor = SamProcessor.from_pretrained(model_id)
    model = SamModel.from_pretrained(model_id).to(device)

    processed_img = processor(img, input_points=boxes, return_tensors="pt").to(device)
    with torch.no_grad(): prediction = model(**processed_img)

    masks = processor.image_processor.post_process_masks(
        prediction.pred_masks.cpu(), 
        processed_img["original_sizes"].cpu(), 
        processed_img["reshaped_input_sizes"].cpu()
    )
    scores = prediction.iou_scores

    return {'masks': masks, 'scores': scores}



    

    


