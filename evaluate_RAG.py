import argparse
import gc
import json
import os
from pathlib import Path

import accelerate
import torch
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import DDIMScheduler
from models.unet import UNet2DConditionModel
from models.AutoencoderKL import AutoencoderKL
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, AutoProcessor
import open_clip
# from xformers.ops import MemoryEfficientAttentionFlashAttentionOp

from dataset.dresscode import DressCodeRetrieval, retrieve_collate_fn
from functools import partial
from models.inversion_adapter import InversionAdapter

from pipes.rag_pipe import FashionRAG_Pipe
from utils.image_from_pipe_rag import generate_images_from_pipe
from utils.set_seeds import set_seed
from utils.val_metrics import compute_metrics

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["WANDB_START_METHOD"] = "thread"

def bytes_to_giga_bytes(bytes):
    return bytes / 1024 / 1024 / 1024

def launch_args():
    with open('.vscode/launch.json', 'r') as f:
        launch_config = json.load(f)

    args = launch_config['configurations'][0].get('args', [])

    return args


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument('--clip_retrieve_model', type=str, default='ViT-L-14') # retrieve_clip_retrieve_model
    parser.add_argument('--clip_retrieve_weights', type=str, default='laion2b_s32b_b82k') # retrieve_clip_retrieve_weights
    parser.add_argument('--retrieve_path', type=str, required=True)
    parser.add_argument('--phase', type=str, default="test")
    parser.add_argument('--n_chunks', type=int, required=True)
    parser.add_argument('--n_retrieved', type=int, default=5)
    parser.add_argument('--compute_metrics', default=False, action="store_true")
    parser.add_argument('--retrieve_mode', type=str, choices=["online", "offline"], default="offline")
    parser.add_argument('--metrics', type=str, nargs="*", required=True)
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument("--unet_name", type=str, default="latest")

    parser.add_argument("--seed", type=int, default=1234, help="A seed for reproducible training.")

    parser.add_argument(
        "--test_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )

    parser.add_argument(
        "--metrics_batch_size", type=int, default=16, help="Batch size (per device) for the testing dataloader."
    )

    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )

    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )

    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    parser.add_argument('--dresscode_dataroot', type=str, help='DressCode dataroot')
    parser.add_argument('--vitonhd_dataroot', type=str, help='VitonHD dataroot')

    parser.add_argument("--num_workers_test", type=int, default=8,
                        help="The name of the repository to keep in sync with the local `output_dir`.",
                        )

    parser.add_argument(
        "--finetuned_models_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--inversion_adapter_name", type=str, default='none')

    parser.add_argument("--test_order", type=str, required=True, choices=["unpaired", "paired"])
    parser.add_argument("--mask_type", type=str, required=True, choices=["mask", "bounding_box"])
    parser.add_argument("--skip_image_extraction", action="store_true")
    parser.add_argument("--dataset", type=str, required=True, choices=["dresscode", "vitonhd"], help="dataset to use")
    parser.add_argument("--no_pose", action="store_true")
    parser.add_argument("--category", type=str, choices=['all', 'lower_body', 'upper_body', 'dresses'], default='all')
    parser.add_argument("--conditioning_mode", choices=['none', 'txt', 'pte', 'pte_text', 'cross-attention'], 
                        type=str, required=False, default='pte')
    parser.add_argument("--merge_modality", choices=['none', 'merge_pte', 'concat_pte'], 
                        type=str, required=False, default='merge_pte')
    parser.add_argument("--text_usage", required=True, choices=['none', 'noun_chunks', 'prompt_noun_chunks'], type=str)
    parser.add_argument("--num_vstar", default=16, type=int)
    parser.add_argument("--num_encoder_layers", default=1, type=int)
    parser.add_argument("--num_inference_steps", default=50, type=int)
    parser.add_argument("--use_retrieved", default=False, action="store_true")
    parser.add_argument("--use_gt_retrieval", default=False, action="store_true")

    parser.add_argument("--attention_layers_fine_list", nargs="+", type=str, default=[0, -1], help="Attention layers fine-tuning list. Use -1 to use all layers. Format: [layer, layer1 layer2 ...]")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="guidance scale")
    parser.add_argument("--retrieved_order", type=str, default='original', choices=["shuffled", "original"])
    parser.add_argument("--generate_htmls", default=False, action="store_true")
    parser.add_argument("--augment_dataset", default=False, action="store_true")
    parser.add_argument("--chunk_list", type=int, nargs="+", default=None)

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1)) # for distributed training
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


@torch.inference_mode()
def main():
    print("parse args")
    args = parse_args()
    # saving utils
    vocab = {"key":10}
    with open(os.path.join(args.output_dir, "vocab.json"), "w") as f:
        json.dump(vocab, f)
    save_name = args.conditioning_mode if args.conditioning_mode != "none" else "no_cond"
    if args.augment_dataset: save_name = save_name + "_aug"
    save_path = os.path.join(args.output_dir, f"{save_name}_{args.test_order}_nc_{args.n_chunks}_nr_{args.n_retrieved}")
    retrieve_root_path = os.path.join(args.retrieve_path)
    retrieve_feat_path = os.path.join(retrieve_root_path, args.clip_retrieve_model + "_" + args.clip_retrieve_weights)
    merge_pte = args.merge_modality == "merge_pte"
    use_embeds = args.conditioning_mode != 'none'  
    use_retrieved = args.n_retrieved > 0
    if merge_pte: args.n_retrieved = min(args.n_retrieved, 77//args.num_vstar)
    if args.chunk_list is not None: assert args.n_chunks == len(args.chunk_list)
    cache_dir = os.environ.get("HF_CACHE_DIR", None)
    print("set args")

    if not args.skip_image_extraction:
        inversion_adapter = None
        attention_layers_fine_dict = {}
        possible_values = [0, 1, 2, 3]
        concat_textual_cond = len(args.attention_layers_fine_list[1].split()) != len(possible_values) and args.n_retrieved > 0 # enable mixed layers conditioning

        # enable only specific layers conditioning if specified
        if args.attention_layers_fine_list[0] == '-1':
            new_attention_layers_fine_list = [0] * (args.num_inference_steps * 2)
            new_attention_layers_fine_list[::2] = list(range(args.num_inference_steps))
            new_attention_layers_fine_list[1::2] =  [args.attention_layers_fine_list[1]] * args.num_inference_steps
            args.attention_layers_fine_list = new_attention_layers_fine_list
            
        for t, layers in zip(args.attention_layers_fine_list[::2], args.attention_layers_fine_list[1::2]):
            t = int(t)
            if len(layers)==1 and int(layers) == -1:
                attention_layers_fine_dict[t] = (int(layers),)
            else:
                attention_layers_fine_dict[t] = tuple(set(map(int, layers.split())))
                for v in attention_layers_fine_dict[t]:
                    assert v in possible_values, f"Attention layer {v} is not valid. Possible values are {possible_values}"

        assert 0 in attention_layers_fine_dict.keys(), "You must specify the attention layer 0"

        if not use_retrieved:
            print('WARNING: not using retrieved images\n' * 20)

        accelerator = Accelerator(
            mixed_precision=args.mixed_precision,
        )

        device = accelerator.device

        # If passed along, set the training seed now.
        if args.seed is not None:
            set_seed(args.seed)

        # Retrieve Modules
        retrieve_model, retrieve_tokenizer = None, None
        if args.retrieve_mode == "online":
            retrieve_model, _, _ = open_clip.create_model_and_transforms(args.clip_retrieve_model, pretrained=args.clip_retrieve_weights)
            retrieve_tokenizer = open_clip.get_tokenizer(args.clip_retrieve_model)
            torch.cuda.empty_cache()
            retrieve_model.eval()

        # Load scheduler, tokenize r and models.
        print("Loading scheduler, tokenizer and models...")
        val_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", cache_dir=cache_dir)
        val_scheduler.set_timesteps(50, device=device)

        text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", cache_dir=cache_dir)

        vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", cache_dir=cache_dir)

        in_feature_channels = [128, 128, 256, 512]
        out_feature_channels = [256, 512, 512, 512]

        int_layers = [2, 3, 4, 5]

        int_layers.sort()

        unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        cache_dir=cache_dir,)

        tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer", cache_dir=cache_dir)

        new_in_channels = 27
        with torch.no_grad():
            # Replace the first conv layer of the unet with a new one with the correct number of input channels
            conv_new = torch.nn.Conv2d(
                in_channels=new_in_channels,
                out_channels=unet.conv_in.out_channels,
                kernel_size=3,
                padding=1,
            )

            # Initialize new conv layer
            torch.nn.init.kaiming_normal_(conv_new.weight)
            conv_new.weight.data = conv_new.weight.data * 0.  # Zero-initialize new conv layer

            # Copy weights from old conv layer
            conv_new.weight.data[:, :9] = unet.conv_in.weight.data
            conv_new.bias.data = unet.conv_in.bias.data  # Copy bias from old conv layer

            unet.conv_in = conv_new  # replace conv layer in unet
            unet.config['in_channels'] = new_in_channels  # update config
            unet.config.in_channels = new_in_channels

        # Freeze vae and text_encoder
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)

        if args.finetuned_models_dir is not None and args.unet_name != "baseline":  # Load unet checkpoint
            try:
                if args.unet_name != "latest":
                    path = args.unet_name
                else:
                    # Get the most recent checkpoint
                    dirs = os.listdir(args.finetuned_models_dir)
                    dirs = [d for d in dirs if d.startswith("unet")]
                    dirs = sorted(dirs, key=lambda x: int(os.path.splitext(x.split("_")[-1])[0]))
                    path = dirs[-1]
                accelerator.print(f"Resuming unet from checkpoint {path}")
                unet.load_state_dict(torch.load(os.path.join(args.finetuned_models_dir, path)))
            except:
                print(f"no checkpoints found in {args.finetuned_models_dir}")
        else:
            print("No unet checkpoint specified\n" * 20, flush=True)

        # if args.enable_xformers_memory_efficient_attention:
        #     if is_xformers_available():
        #         unet.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
        #     else:
        #         raise ValueError("xformers is not available. Make sure it is installed correctly")

        if args.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        outputlist = ['image', 'cloth', 'pose_map', 'captions', 'inpaint_mask', 'im_mask', 'category', 'im_name', 'c_name']
        retrievelist = []

        if use_retrieved:
            retrievelist.extend(['retrieved_cloth', 'retrieved_cpaths'])
            # possible loading problems
            if args.pretrained_model_name_or_path == "runwayml/stable-diffusion-inpainting":
                new_weights = "openai/clip-vit-large-patch14"
                vision_encoder = CLIPVisionModelWithProjection.from_pretrained(
                    new_weights)
                processor = AutoProcessor.from_pretrained(
                    new_weights)
            elif args.pretrained_model_name_or_path == "stabilityai/stable-diffusion-2-inpainting":
                vision_encoder = CLIPVisionModelWithProjection.from_pretrained(
                    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
                processor = AutoProcessor.from_pretrained(
                    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
            else:
                raise ValueError(
                    f"Unknown pretrained model name or path: {args.pretrained_model_name_or_path}")

            vision_encoder.requires_grad_(False)
            vision_encoder.eval()

            # Inversion adapter pre-trained weights dimensions must be compatible with selected vision encoder dimensions
            inversion_adapter = InversionAdapter(input_dim=vision_encoder.config.hidden_size,
                                                hidden_dim=vision_encoder.config.hidden_size * 4,
                                                output_dim=text_encoder.config.hidden_size * args.num_vstar,
                                                num_encoder_layers=args.num_encoder_layers,
                                                config=vision_encoder.config)

            if args.finetuned_models_dir is not None:  # Load inversion adapter checkpoint
                if args.inversion_adapter_name != "latest":
                    path = args.inversion_adapter_name
                else:
                    # Get the most recent checkpoint
                    dirs = os.listdir(args.finetuned_models_dir)
                    dirs = [d for d in dirs if d.startswith("inversion_adapter")]
                    dirs = sorted(dirs, key=lambda x: int(
                        os.path.splitext(x.split("_")[-1])[0]))
                    path = dirs[-1]
                accelerator.print(f"Resuming from checkpoint {path}")
                inversion_adapter.load_state_dict(torch.load(
                    os.path.join(args.finetuned_models_dir, path)))
            else:
                print("No inversion adapter checkpoint directory specified. Using random initialized module")
        else:
            vision_encoder = None
            processor = None
            inversion_adapter = None
            print("YOU'RE NOT USING THE RETRIEVED IMAGES\n" * 20, flush=True)

        if args.category != 'all':
            category = [args.category]
        else:
            category = ['dresses', 'upper_body', 'lower_body']

        if args.dataset == "dresscode":
            test_dataset = DressCodeRetrieval(
                dataroot_path=args.dresscode_dataroot,
                phase=args.phase,
                order=args.test_order,
                radius=5,
                outputlist=outputlist,
                sketch_threshold_range=(20, 20),
                category=category,
                size=(512, 384),
                retrieve_feat_path=retrieve_feat_path,
                n_chunks=args.n_chunks,
                top_k=args.n_retrieved,
                augment_dataset = args.augment_dataset
                )
        else:
            raise NotImplementedError(f"Dataset {args.dataset} not implemented")

        collate_fn = partial(retrieve_collate_fn, var_keys=retrievelist, n_retrieved=args.n_retrieved)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=args.test_batch_size,
            num_workers=args.num_workers_test,
            collate_fn=collate_fn
        )

        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        # Move text_encode and vae to gpu and cast to weight_dtype
        text_encoder.to(device, dtype=weight_dtype)
        vae.to(device, dtype=weight_dtype)
        unet.to(device, dtype=weight_dtype)

        if inversion_adapter is not None:
            inversion_adapter.to(device, dtype=weight_dtype)

        if vision_encoder is not None:
            vision_encoder.to(device, dtype=weight_dtype)
        # Move text_encode and vae to gpu and cast to weight_dtype

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.

        unet.eval()
        if inversion_adapter is not None:
            inversion_adapter.eval()
        with torch.inference_mode():
            val_pipe = FashionRAG_Pipe(
                text_encoder=text_encoder,
                vae=vae,
                unet=unet,
                tokenizer=tokenizer,
                scheduler=val_scheduler,
                safety_checker=None,
                requires_safety_checker=False,
                feature_extractor=None,
            ).to(accelerator.device)

            # val_pipe.enable_attention_slicing()
            test_dataloader = accelerator.prepare(test_dataloader)
            os.makedirs(save_path, exist_ok=True)
                    
            generate_images_from_pipe(pipe=val_pipe,
                                            inversion_adapter=inversion_adapter,
                                            test_dataloader=test_dataloader,
                                            save_path=save_path,
                                            vision_encoder=vision_encoder,
                                            processor=processor,
                                            num_vstar=args.num_vstar,
                                            seed=args.seed,
                                            num_inference_steps=args.num_inference_steps,
                                            guidance_scale=args.guidance_scale,
                                            merge_pte = merge_pte,
                                            n_retrieved = args.n_retrieved,
                                            text_usage = args.text_usage,
                                            use_embeds = use_embeds,
                                            attention_layers_fine_dict=attention_layers_fine_dict,
                                            concat_textual_cond=concat_textual_cond,
                                            use_retrieved = args.n_retrieved > 0,
                                            use_chunks = True
                                            )

            val_pipe = None
            vision_encoder = None
            processor = None
            inversion_adapter = None
            text_encoder = None
            vae = None
            unet = None

            gc.collect()
            torch.cuda.empty_cache()

    
    use_chunks = True
    if args.compute_metrics:
        # args.category = "all"
        metrics_name = f"metrics_{args.category}.json"

        metrics = compute_metrics(
            save_path,
            args.test_order, args.dataset, args.category,
            args.metrics, args.dresscode_dataroot,
            args.vitonhd_dataroot, batch_size=args.metrics_batch_size, n_chunks=args.n_chunks, n_retrieved=args.n_retrieved, use_chunks=use_chunks, 
            retrieve_path=retrieve_feat_path, aug_captions=args.augment_dataset)
        
        metrics = {f"chunks_{args.n_chunks}_retrieved_{args.n_retrieved}": metrics}
        try:
            with open(os.path.join(save_path,
                                metrics_name),
                    "r") as f:
                    try: existing_data = json.load(f)
                    except:  existing_data = dict({})
                    metrics.update(existing_data)
        except:
            pass

        metrics = dict(sorted(metrics.items()))
        with open(os.path.join(save_path,
                                metrics_name),
                "w+") as f:
            json.dump(metrics, f, indent=4) # put sorted here

        with open(os.path.join(save_path,
                            f'hyperparameters_{args.category}.json'),
                'w+') as f:
            json.dump(vars(args), f, sort_keys=True, indent=4)


if __name__ == "__main__":
    main()