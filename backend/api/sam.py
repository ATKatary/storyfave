import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import SamModel, SamProcessor

def get_masks(img, boxes, model="sam", device="cuda"):
    """ Runs SAM to get the predicted masks and scores """
    if len(boxes[0]) == 0: return 

    model_id = "facebook/sam-vit-huge"
    processor = SamProcessor.from_pretrained(model_id)
    model = SamModel.from_pretrained(model_id).to(device)

    processed_img = processor(img, input_boxes=[boxes], return_tensors="pt").to(device)
    img_embeddings = model.get_image_embeddings(processed_img["pixel_values"])
    
    processed_img.pop("pixel_values", None)
    processed_img.update({"image_embeddings": img_embeddings})

    with torch.no_grad(): prediction = model(**processed_img)

    masks = processor.image_processor.post_process_masks(
        prediction.pred_masks.cpu(), 
        processed_img["original_sizes"].cpu(), 
        processed_img["reshaped_input_sizes"].cpu()
    )
    scores = prediction.iou_scores

    return {'masks': masks, 'scores': scores}

def show_mask(mask, ax, random_color=False):
    """ Displays a predicted mask on the image """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
        
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_masks_on_image(raw_image, masks, scores):
    """ Displays all predicted masks on the image """
    if len(masks.shape) == 4: masks = masks.squeeze()
    if scores.shape[0] == 1: scores = scores.squeeze()

    nb_predictions = scores.shape[-1]
    fig, axes = plt.subplots(1, nb_predictions, figsize=(15, 15))

    for i, (mask, score) in enumerate(zip(masks, scores)):
      mask = mask.cpu().detach()
      axes[i].imshow(np.array(raw_image))

      show_mask(mask, axes[i])
      axes[i].title.set_text(f"Mask {i+1}, Score: {score.item():.3f}")
      axes[i].axis("off")
    plt.show()
