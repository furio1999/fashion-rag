import os, json
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Literal
from tqdm import tqdm

import torch
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor, CLIPProcessor
from pipes.rag_pipe import FashionRAG_Pipe
from models.inversion_adapter import InversionAdapter
from utils.encode_text_word_embedding import encode_text_word_embedding

from PIL import Image

PROJECT_ROOT = Path(__file__).absolute().parents[2].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

@torch.no_grad()
def generate_images_from_pipe(pipe: FashionRAG_Pipe,
                                    inversion_adapter: InversionAdapter,
                                    test_dataloader: torch.utils.data.DataLoader,
                                    save_path: str,
                                    vision_encoder: CLIPVisionModelWithProjection,
                                    processor: CLIPProcessor,
                                    attention_layers_fine_dict: dict[tuple[int, tuple]],
                                    sketch_cond_rate: int = 1,
                                    num_vstar: int = 1,
                                    seed: int = 1234,
                                    num_inference_steps: int = 50,
                                    guidance_scale: int = 7.5,
                                    text_usage: Literal['prompt_noun_chunks', 'noun_chunks', 'none'] = "noun_chunks",
                                    concat_textual_cond: bool = False,
                                    use_retrieved : bool = False,
                                    attn_processor=None,
                                    n_retrieved=1, merge_pte=True, use_embeds=True,
                                    use_chunks=False
                                    ) -> None:
    """_summary_

    Args:
        pipe (FashionRAG_Pipe): validation pipeline
        inversion_adapter (InversionAdapter): Inversion adapter neetwork
        test_dataloader (torch.utils.data.DataLoader): torch.dataloader for validation dataset
        output_dir (str): output directory,  and output images will be saved here
        order (str): paired unpaired dataset setting
        save_name (str): name to save the output images
        text_usage (bool): use noun chunks or not
        vision_encoder (CLIPVisionModelWithProjection): _description_
        processor (CLIPProcessor): _description_
        sketch_cond_rate (int, optional): _description_. Defaults to 1.
        num_vstar (int, optional): _description_. Defaults to 1.
        seed (int, optional): _description_. Defaults to 1234.
        num_inference_steps (int, optional): _description_. Defaults to 50.
        guidance_scale (int, optional): _description_. Defaults to 7.5.
        use_png (bool, optional): _description_. Defaults to False.
        use_retrieved (bool, optional): _description_. Defaults to False.
        attention_layers_fine_dict (dict[tuple[int, tuple]]): timesteps and layers in which perform retrieved images inversion attention,
                {t: (0, 1, 2, 3)}. t = conditioning timestep. 
                0 = mid layer, 1 = coarse layers (3 up and 1 down), 2 = (2 up and 2 down), 3 = finest layers (1 up and 3 down) 3 is the outest Defaults to tuple().

    """
    if concat_textual_cond:
        assert use_retrieved, "concat_textual_cond is only available when use_retrieved is True"
    else:
        if use_retrieved:
            for _, v in attention_layers_fine_dict.items():
                if v[0] != -1:
                    break
            else:
                raise ValueError(
                    "At least one of the attention attention_layers_fine_dict must be specified when use_retrieved is True")

    # Create output directory
    # save_path = os.path.join(output_dir, f"{save_name}_{order}_{texture_order}")
    os.makedirs(save_path, exist_ok=True)

    dataroot = test_dataloader.dataset.dataroot # check if all classes have this feat
    n_chunks, phase = test_dataloader.dataset.n_chunks, test_dataloader.dataset.phase # "single", "train" # extract from test_dataloader
    combos = ["single", "double", "triplet"]
    combo_name = combos[n_chunks-1]

    generator = torch.Generator("cuda").manual_seed(seed)
    num_samples = 1

    vocab = {}
    for idx, batch in enumerate(tqdm(test_dataloader)):
        model_img = batch.get("image").to(pipe.device)
        mask_img = batch.get("inpaint_mask")
        if mask_img is not None:
            mask_img = mask_img.type(torch.float32)
        pose_map = batch.get("pose_map")
        category = batch.get("category")
        sketch = batch.get("im_sketch")
        bs = len(category)

        retrieved_paths = bs * [[]]
        if "retrieved_cloth" in batch.keys():
            # retrieval part
            retrieved_images = batch["retrieved_cloth"]
            retrieved_paths = batch["retrieved_cpaths"]
            r_index = torch.Tensor([bi*n_retrieved + j for bi, batch_ret in enumerate(retrieved_paths) for j in range(len(batch_ret), n_retrieved)]).to(torch.int64)
            nr_per_item = [len(batch_ret) for batch_ret in retrieved_paths]
            tot_pte = num_vstar * sum(nr_per_item)

        category_text = {
            'dresses': 'a dress',
            'upper_body': 'an upper body garment',
            'lower_body': 'a lower body garment',
        }

        encoder_hidden_states = None
        encoder_hidden_states_fine = None
        word_embeddings = None

        if use_retrieved:

            input_image = (retrieved_images + 1) / 2  # Scale to [0, 1]
            processed_images = processor(images=list(input_image), return_tensors="pt") # 3x224x224
            clip_cloth_features = vision_encoder(processed_images.pixel_values.to(
                model_img.device).to(vision_encoder.dtype)).last_hidden_state # 1024

            if merge_pte: # merge_pte
                word_embeddings = inversion_adapter(clip_cloth_features.to(model_img.device))
                word_embeddings = word_embeddings.reshape((len(category), num_vstar*n_retrieved, -1)) # ensured to have concatenated feats?
            else:
                word_embeddings = inversion_adapter(clip_cloth_features.to(model_img.device)) # call n_retrievd times?
                word_embeddings = word_embeddings.reshape((len(category)*n_retrieved, num_vstar, -1))             


        category_text = {
            'dresses': 'a dress',
            'upper_body': 'an upper body garment',
            'lower_body': 'a lower body garment',

        }

        if merge_pte and use_retrieved and text_usage == "prompt_noun_chunks": # if mode == "merge_pte"
            fine_text = [f'a photo of a model wearing {category_text[batch["category"][n]]} made of {" $ " * num_vstar * n_retrieved}'
                    for n, n_retrieved in enumerate(nr_per_item)]
        elif merge_pte and use_retrieved and text_usage == "noun_chunks":
            fine_text = [f'{batch["captions"][n]}. {" $ " * num_vstar * n_retrieved}' for n, n_retrieved in enumerate(nr_per_item)]
        elif use_retrieved and text_usage == "prompt_noun_chunks": # mode == "concat_pte"
            fine_text = [f'a photo of a model wearing {category_text[batch["category"][n//n_retrieved]]} made of {" $ " * num_vstar}'
                    for n in range(len(category)*n_retrieved)]
        elif use_retrieved and text_usage == "noun_chunks":
            fine_text = [f'{batch["captions"][n//n_retrieved]}. {" $ " * num_vstar}' for n in range(len(category)*n_retrieved)]
        else:
            fine_text = [""] * len(batch["captions"])

        if text_usage == "prompt_noun_chunks":
            text = [f'a photo of a model wearing {category_text[batch["category"][n]]}. {batch["captions"][n]}' for
                n in range(len(category))] # bs*n_retrieved correct here?
        elif text_usage == "noun_chunks":
            text = [f'{batch["captions"][n]}' for n in range(len(category))]
        elif text_usage == "none":
            text = [""] * len(batch["captions"])
        else:
            raise ValueError(f"Unknown text usage {text_usage}")


        # Tokenize text [N, tokenizer.model_max_length=77]
        tokenized_text = pipe.tokenizer(text, max_length=pipe.tokenizer.model_max_length, padding="max_length",
                                        truncation=True, return_tensors="pt").input_ids 
        tokenized_text = tokenized_text.to(model_img.device)
        if use_retrieved:
            tokenized_text_fine = pipe.tokenizer(fine_text, max_length=pipe.tokenizer.model_max_length, padding="max_length",
                                    truncation=True, return_tensors="pt").input_ids
            tokenized_text_fine = tokenized_text_fine.to(model_img.device)    
            
        if use_embeds:
            text = None
            return_pte = attn_processor is not None

            # Encode the text using the PTEs extracted from the in-shop cloths
            # both True means using both pte and text. both false means text only. use_retrieved True ans concat_text_cond False means ptes only
            if merge_pte and concat_textual_cond == use_retrieved:
                encoder_hidden_states = encode_text_word_embedding(pipe.text_encoder, tokenized_text,
                                                                word_embeddings, n_retrieved=None,
                                                                num_vstar=num_vstar*n_retrieved, return_pte=return_pte)

            if merge_pte and use_retrieved:
                encoder_hidden_states_fine = encode_text_word_embedding(pipe.text_encoder, tokenized_text_fine,
                                                                word_embeddings, n_retrieved=None,
                                                                num_vstar=num_vstar*n_retrieved, return_pte=return_pte)

            # if no encoder_hidden_states assign the PTEs to all the layers    
            if encoder_hidden_states is None and encoder_hidden_states_fine is not None:
                encoder_hidden_states = encoder_hidden_states_fine.clone() # to avoid in place assignments outside torch.InferenceMode
                encoder_hidden_states_fine = None

        # Generate images
        generated_images = pipe(
            image=model_img,
            mask_image=mask_img,
            pose_map=pose_map,
            prompt_embeds=encoder_hidden_states, # 77x1024
            prompt_embeds_fine=encoder_hidden_states_fine, # TODO
            attention_layers_fine_dict=attention_layers_fine_dict,
            height=512,
            width=384,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_samples,
            generator=generator,
            sketch=sketch,
            sketch_cond_rate=sketch_cond_rate,
            num_inference_steps=num_inference_steps
        ).images

        # Handle text if is not None
        # Save images
        use_chunks = use_chunks # save_cond_name = "chunks"
        for i, (caption, gen_image, cat, im_name, c_name, rpaths) in enumerate(zip(batch["captions"], generated_images, category, batch["im_name"], batch['c_name'], retrieved_paths)):
            
            if not os.path.exists(os.path.join(save_path, cat)):
                os.makedirs(os.path.join(save_path, cat))
            
            save_caption = "_".join(["_".join(chunk.split(" ")) for chunk in caption.split(", ")]).replace("/", "_")
            if use_chunks: name = im_name.replace(".jpg", f"_{save_caption}.jpg")
            else: name = im_name.replace(".jpg", f"_{c_name}")
            name = name.replace(".jpg", ".png")
            gen_image.save(
                os.path.join(save_path, cat, name))
            print("image saved in: ", os.path.join(save_path, cat, name))
