import torch
import numpy as np
from rembg import remove
from storyfave.backend.utils import *
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler, ControlNetModel

class MultiViewer(torch.nn.Module):
    def __init__(self, use_controlnet=False, dtype = torch.float16):
        super(MultiViewer, self).__init__()
        self.dtype = dtype

        self.pipeline = DiffusionPipeline.from_pretrained(ZERO_PLUS, **ZERO_PLUS_CONFIG, torch_dtype=dtype)

        if use_controlnet:
            self.controlnet = ControlNetModel.from_pretrained(ZERO_PLUS_CONTROLNET, torch_dtype=dtype)
            self.pipeline.add_controlnet(self.controlnet, conditioning_scale=0.75)

        self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipeline.scheduler.config, timestep_spacing='trailing')

    def __call__(self, x, **kwargs):
        return self.pipeline(x, **kwargs).images[0]

    def to(self, device):
        self.pipeline.to(device)
    
def split_char_sheet(img, k_w=2, k_h=3):
    img = np.array(img)

    h, w, c = img.shape
    n, m = h//k_h, w//k_w

    return [
        remove(img[i:i + n, j:j + m, :])
        for j in range(0, w, m) for i in range(0, h, n)
    ]