import os
import argparse

def retrieval_parser():
    parser = argparse.ArgumentParser(
    description="Retrieval script")

    parser.add_argument(
    "--top_k",
    type=int,
    default=5,
    required=False,
    help="Top K images to be extracted from a text description",
)

    args = parser.parse_args()
    return args

def train_rag_parse_args():
    parser = argparse.ArgumentParser(
        description="Simple example of a training script.")
    
    parser.add_argument("--force_cpu", action="store_true", default=False)

    parser.add_argument("--pin_memory", action="store_true", default=False)

    parser.add_argument("--shuffle_list", nargs='*', choices=["none", "retrieved", "captions"], type=str, default=[]) # '+' requires at least one

    parser.add_argument("--we_type", type=str, default="tensor")

    parser.add_argument("--use_crossattn", default=False, action="store_true")

    parser.add_argument("--conditioning_mode", choices=['none', 'pte', 'cross-attention'], 
                        type=str, required=True, default='pte')
    parser.add_argument("--conditioning_data", choices=['retrieved', 'texture', 'cloth'], 
                        type=str, required=True)
    
    parser.add_argument("--merge_modality", choices=['none', 'merge_pte', 'concat_pte'], 
                        type=str, required=True, default='merge_pte')

    parser.add_argument('--n_chunks', type=int, required=True)
    parser.add_argument('--n_retrieved', type=int, default=5, required=True)
    parser.add_argument('--retrieve_path', type=str, required=True)
    parser.add_argument('--clip_retrieve_model', type=str, default='ViT-L-14') # retrieve_clip_model
    parser.add_argument('--clip_retrieve_weights', type=str, default='laion2b_s32b_b82k') # retrieve_clip_weights
    parser.add_argument('--plot_qualitatives', action="store_true", default=False) # retrieve_clip_weights

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument("--seed", type=int, default=1234,
                        help="A seed for reproducible training.")

    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )

    parser.add_argument(
        "--test_batch_size", type=int, default=16, help="Batch size (per device) for the testing dataloader."
    )

    parser.add_argument(
        "--metrics_batch_size", type=int, default=16, help="Batch size (per device) for the testing dataloader."
    )

    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )

    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

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
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )

    parser.add_argument("--inversion_adapter_dir", type=str, default=None,
                        help="Path to the directory containing the inversion adapter.",
                        )
    parser.add_argument("--inversion_adapter_name", type=str, default="latest")

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--final_checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Final value of checkpointing_steps to enable a more frequent evaluation to avoid overfitting."
        ),
    )
    parser.add_argument(
        "--t_start_rise",
        type=int,
        default=75,
        metavar="[0-100]",
        help=(
            "percentange of timesteps after which increasing the number of evaluations. Defaults to 75% of total training steps"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument('--cache_dir', type=str, default=None)

    parser.add_argument('--dresscode_dataroot', type=str, help='DressCode dataroot')
    parser.add_argument('--vitonhd_dataroot', type=str, help='VitonHD dataroot')

    parser.add_argument("--num_workers", type=int, default=8,
                        help="The name of the repository to keep in sync with the local `output_dir`.", )

    parser.add_argument("--num_workers_test", type=int, default=8,
                        help="The name of the repository to keep in sync with the local `output_dir`.", )

    parser.add_argument("--test_order", type=str, default="unpaired", choices=["unpaired", "paired"])
    parser.add_argument("--uncond_fraction", type=float, default=0.0,
                        help="Fraction of unconditioned training samples")

    parser.add_argument("--mask_type", type=str, required=True, choices=["mask", "bounding_box"])
    parser.add_argument("--new_weights_order", type=float, default=0, help="new weights downscaling exponent")
    parser.add_argument("--dataset", type=str, required=True, choices=["dresscode", "vitonhd"], help="dataset to use")
    parser.add_argument("--num_vstar", default=1, type=int)
    parser.add_argument("--num_query_tokens", default=32, type=int, help='number of learnable embeddings in Q-Former')
    parser.add_argument("--num_encoder_layers", default=1, type=int)
    parser.add_argument("--train_inversion_adapter", default=False, action="store_true")
    parser.add_argument("--train_qformer", default=False, action="store_true")
    parser.add_argument("--text_usage", default='prompt_noun_chunks', choices=['none', 'noun_chunks', 'prompt_noun_chunks'],
                        type=str)
    parser.add_argument("--text_encoder_hidden_size", type=int, default=1024, help="text encoder hidden dimension")
    parser.add_argument("--use_conditioning", default=False, action="store_true")
    parser.add_argument("--fine_grained_retrieved", default=False, action="store_true")
    parser.add_argument("--use_sketch", default=False, action="store_true")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="guidance scale")
    parser.add_argument("--sketch_cond_rate", type=float, default=0.2, help="sketch conditioning rate")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="number of inference steps")
    parser.add_argument("--noun_chunks_rate", type=float, default=0.5,
                        help="noun chunks rate vs retrieved images rate. When is 1 only noun chunks are used, when is 0 only"
                             "retrieved images are used")

    parser.add_argument("--retrieved_order", type=str, default='original', choices=["shuffled", "original"])

    parser.add_argument("--generate_html", default=False, action="store_true")


    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args



def train_mgd_parse_args():
    parser = argparse.ArgumentParser(
        description="Simple example of a training script.")

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument('--cache_dir', type=str, default=None)
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument("--seed", type=int, default=1234,
                        help="A seed for reproducible training.")

    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )

    parser.add_argument(
        "--test_batch_size", type=int, default=16, help="Batch size (per device) for the testing dataloader."
    )

    parser.add_argument(
        "--metrics_batch_size", type=int, default=16, help="Batch size (per device) for the testing dataloader."
    )

    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )

    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

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
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )

    parser.add_argument("--inversion_adapter_dir", type=str, default=None,
                        help="Path to the directory containing the inversion adapter.",
                        )
    parser.add_argument("--inversion_adapter_name", type=str, default="latest")

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    parser.add_argument('--dresscode_dataroot', type=str, help='DressCode dataroot')
    parser.add_argument('--vitonhd_dataroot', type=str, help='VitonHD dataroot')

    parser.add_argument("--num_workers", type=int, default=8,
                        help="The name of the repository to keep in sync with the local `output_dir`.", )

    parser.add_argument("--num_workers_test", type=int, default=8,
                        help="The name of the repository to keep in sync with the local `output_dir`.", )

    parser.add_argument("--test_order", type=str, default="unpaired", choices=["unpaired", "paired"])
    parser.add_argument("--uncond_fraction", type=float, default=0.0,
                        help="Fraction of unconditioned training samples")

    parser.add_argument("--mask_type", type=str, required=True, choices=["mask", "bounding_box"])
    parser.add_argument("--new_weights_order", type=float, default=0, help="new weights downscaling exponent")
    parser.add_argument("--dataset", type=str, required=True, choices=["dresscode", "vitonhd"], help="dataset to use")
    parser.add_argument("--num_vstar", default=1, type=int)
    parser.add_argument("--num_encoder_layers", default=1, type=int)
    parser.add_argument("--train_inversion_adapter", default=False, action="store_true")
    parser.add_argument("--text_usage", default='prompt_noun_chunks', choices=['none', 'noun_chunks', 'prompt_noun_chunks'],
                        type=str)
    parser.add_argument("--use_texture", default=False, action="store_true")
    parser.add_argument("--fine_grained_texture", default=False, action="store_true")
    parser.add_argument("--use_sketch", default=False, action="store_true")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="guidance scale")
    parser.add_argument("--sketch_cond_rate", type=float, default=0.2, help="sketch conditioning rate")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="number of inference steps")
    parser.add_argument("--noun_chunks_rate", type=float, default=0.5,
                        help="noun chunks rate vs texture rate. When is 1 only noun chunks are used, when is 0 only"
                             "texture is used")

    parser.add_argument("--texture_order", type=str, default='original', choices=["shuffled", "original"])


    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def train_invadapter_over_mgd_args():
    parser = argparse.ArgumentParser(
        description="Simple example of a training script.")

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # parser.add_argument(
    #     "--unet_dir",
    #     type=str,
    #     required=True,
    #     help="The output directory where the model predictions and checkpoints will be written.",
    # )

    # parser.add_argument("--unet_name", type=str, default="latest")


    parser.add_argument("--seed", type=int, default=1234,
                        help="A seed for reproducible training.")

    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )

    parser.add_argument(
        "--test_batch_size", type=int, default=16, help="Batch size (per device) for the testing dataloader."
    )

    parser.add_argument(
        "--metrics_batch_size", type=int, default=16, help="Batch size (per device) for the testing dataloader."
    )

    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )

    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

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
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )

    parser.add_argument("--inversion_adapter_dir", type=str, default=None,
                        help="Path to the directory containing the inversion adapter.",
                        )
    parser.add_argument("--inversion_adapter_name", type=str, default="latest")

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    parser.add_argument('--dresscode_dataroot', type=str, help='DressCode dataroot')
    parser.add_argument('--vitonhd_dataroot', type=str, help='VitonHD dataroot')

    parser.add_argument("--num_workers", type=int, default=8,
                        help="The name of the repository to keep in sync with the local `output_dir`.", )

    parser.add_argument("--num_workers_test", type=int, default=8,
                        help="The name of the repository to keep in sync with the local `output_dir`.", )

    parser.add_argument("--test_order", type=str, default="unpaired", choices=["unpaired", "paired"])
    parser.add_argument("--uncond_fraction", type=float, default=0.0,
                        help="Fraction of unconditioned training samples")

    parser.add_argument("--mask_type", type=str, required=True, choices=["mask", "bounding_box"])
    parser.add_argument("--dataset", type=str, required=True, choices=["dresscode", "vitonhd"], help="dataset to use")
    parser.add_argument("--num_vstar", default=1, type=int)
    parser.add_argument("--num_encoder_layers", default=1, type=int)
    parser.add_argument("--use_sketch", default=False, action="store_true")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="guidance scale")
    parser.add_argument("--sketch_cond_rate", type=float, default=0.2, help="sketch conditioning rate")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="number of inference steps")

    parser.add_argument("--texture_order", type=str, default='original', choices=["shuffled", "original"])

    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args
