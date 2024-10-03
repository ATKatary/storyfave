import torch
from PIL import Image
import requests
from transformers import SamModel, SamProcessor

model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
input_points = [[[450, 600]]]  # 2D location of a window in the image

inputs = processor(raw_image, input_points=input_points, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

masks = processor.image_processor.post_process_masks(
    outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
)
scores = outputs.iou_scores

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



    

    


