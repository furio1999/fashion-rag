import os.path
from typing import Literal

from transformers import CLIPVisionModelWithProjection, CLIPTextModel, AutoConfig

dependencies = ['torch', 'diffusers', 'transformers']
import torch
from diffusers import UNet2DConditionModel
from models.inversion_adapter import InversionAdapter


def load_inversion_adapter(dataset: Literal['dresscode', 'vitonhd']):
    config = AutoConfig.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    text_encoder_config =  UNet2DConditionModel.load_config("stabilityai/stable-diffusion-2-inpainting", subfolder="text_encoder")
    inversion_adapter = InversionAdapter(input_dim=config.vision_config.hidden_size,
                                         hidden_dim=config.vision_config.hidden_size * 4,
                                         output_dim=text_encoder_config['hidden_size'] * 16,
                                         num_encoder_layers=1,
                                         config=config.vision_config)

    checkpoint_url = f"https://github.com/miccunifi/ladi-vton/releases/download/weights/inversion_adapter_{dataset}.pth"
    inversion_adapter.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint_url, map_location='cpu'))
    return inversion_adapter