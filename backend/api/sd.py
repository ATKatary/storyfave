import torch
from tqdm import tqdm
import torch.nn.functional as F
from storyfave.backend.utils import *
from transformers import CLIPTextModel
from storyfave.backend.api.lora import *
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, StableDiffusionPipeline

class SD(torch.nn.Module):

    def __init__(self, model_id=SDV1_5, tokenizer=default_tokenizer, dtype = torch.float32):
        super(SD, self).__init__()
        self.dtype = dtype
        self.model_id = model_id

        self.tokenizer = tokenizer
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
        self.noise_scheduler = DDPMScheduler.from_config(model_id, subfolder="scheduler")
        self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
        

    def forward(self, x):
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        latent_x = self.vae.encode(x['img'].to(self.vae.device, dtype=self.dtype)).latent_dist.sample() * 0.18215
        noise = torch.randn_like(latent_x)

        low, high = 0, self.noise_scheduler.config.num_train_timesteps
        timesteps = torch.randint(low, high, latent_x.shape[:1], device=latent_x.device).long()

        noisy_latent_x = self.noise_scheduler.add_noise(latent_x, noise, timesteps)
        encoded_prompt = self.text_encoder(x["prompt"].to(self.vae.device))[0]

        return self.unet(noisy_latent_x, timesteps, encoded_prompt).sample, noise

    def to(self, device):
        self.vae.to(device, dtype=self.dtype)
        self.unet.to(device, dtype=self.dtype)
        self.text_encoder.to(device, dtype=self.dtype)
    
    def save(self, accelerator, out_fn="lora_weight"):
        pipeline = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            unet=accelerator.unwrap_model(self.unet),
            text_encoder=accelerator.unwrap_model(self.text_encoder),
        )
        loras = {
            "unet": (pipeline.unet, DEFAULT_TARGET_REPLACE)
        }

        save_safeloras(loras, f"{BASE_DIR}/saved/{out_fn}.safetensors")
        save_lora_weight(pipeline.unet, f"{BASE_DIR}/saved/{out_fn}.pt")

def train_epoch(sd, data_loader, accelerator, optimizer, lr_scheduler, progress_bar):
    for i, batch in enumerate(data_loader):
        pred, noise = sd(batch)

        loss = F.mse_loss(pred.float(), noise.float(), reduction="mean")

        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        progress_bar.update(1)
        progress_bar.set_postfix({"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]})