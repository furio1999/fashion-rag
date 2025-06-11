<!-- PROJECT LOGO -->
<br />
<div align="center">
   <h3 align="center">Fashion-RAG</h3>
      <p align="center"> Fashion-RAG: Multimodal Fashion Image Editing via Retrieval-Augmented Generation</p>
</div>

<div align="center">
  <picture>
    <source srcset="assets/teaser.png" media="(prefers-color-scheme: dark)">
      <img src="assets/teaser.png" width="75%" alt="MiniMax">
    </source>
  </picture>
</div>

<div align="center">

> **Fashion-RAG: Multimodal Fashion Image Editing via Retrieval-Augmented Generation** 
> <p align="center">
  <strong>International Joint Conference on Neural Networks (IJCNN) 2025<br>Oral Presentation</strong>
</p>

> [Fulvio Sanguigni](https://scholar.google.com/citations?user=tSpzMUEAAAAJ&hl=en)<sup>1,2,\*</sup>, [Davide Morelli](https://scholar.google.com/citations?hl=en&user=UJ4D3rYAAAAJ&view_op=list_works)<sup>1,2,\*</sup>, [Marcella Cornia](https://scholar.google.com/citations?user=DzgmSJEAAAAJ&hl=en)<sup>1</sup>, [Rita Cucchiara](https://scholar.google.com/citations?user=OM3sZEoAAAAJ&hl=en)<sup>1</sup>   
> <sup>1</sup>University of Modena, <sup>2</sup>University of Pisa
</div>

<div align="center">
  <a href="https://arxiv.org/abs/2504.14011" style="margin: 0 2px;">
    <img src="https://img.shields.io/badge/Paper-Arxiv-darkred.svg" alt="Paper">
  </a>
  <a href="https://arxiv.org/pdf/2504.14011" style="margin: 0 2px;">
    <img src="https://img.shields.io/badge/PDF-Arxiv-darkred.svg" alt="PDF">
  </a>
  <a href='https://fashion-rag-page.github.io/' style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/Webpage-Project-silver?style=flat&logo=&logoColor=orange' alt='webpage'>
  </a>
  <a href="https://huggingface.co/furio19/fashion-rag">
    <img src="https://img.shields.io/badge/HuggingFace-Model-FFB000.svg" alt="Project">
  </a>
  <a href="https://raw.githubusercontent.com/furio1999/fashion-rag/refs/heads/main/LICENSE?token=GHSAT0AAAAAACZM6UVFACIVYIJVXCSFT2VA2CJR5HQ" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/License-CC BY--NC 4.0-lightgreen?style=flat&logo=Lisence' alt='License'>
  </a>
</div>

<!-- TABLE OF CONTENTS -->
<details>
<summary>Table of Contents</summary>
<ol>
  <li><a href="#about-the-project">About The Project</a></li>
  <li><a href="#getting-started">Getting Started</a>
    <ul>
      <li><a href="#prerequisites">Prerequisites</a></li>
      <li><a href="#installation">Installation</a></li>
    </ul>
  </li>
  <li><a href="#inference">Inference</a></li>
</ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

Fashion-RAG is a novel approach in the fashion domain, handling multimodal virtual dressing with a new, Retrieval Augmented Generation (RAG) pipeline for visual data. 
Our approach allows to retrieve garments aligned with a given textual description, and uses the retrieved information as a conditioning to generate the dressed person with Stable Diffusion (SD) as the generative model. We finetune the SD U-Net and an additional adapter module (Inversion Adapter) to handle for the retrieved information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## ✨ Key Features
Our contribution can be summarized as follows:
- **🔍 Retrieval Enhanced Generation for Visual Items**. We present a unified framework capable of generating Virtual Dressing without the need of a user-defined garment image.
Instead, our method succesfully leverages textual information and retrieves coherent garments to perform the task
- **👗👚🧥 Multiple Garments Conditioning**. We introduce a plug-and-play adapter module that is flexible to the number of retrieved items, allowing to retrieve up to 3 garments per text prompt.
- **📊 Extensive experiments**. Experiments on the Dress Code datasets demonstrate that Fahion-RAG outweights previous competitors.

<!-- GETTING STARTED -->
## Getting Started
### Prerequisites

Clone the repository:
  ```sh
  git clone Fashion-RAG.git
  ```

### Installation

1. We recommend installing the required packages using Python's native virtual environment (venv) as follows:
   ```sh
   python -m venv fashion-rag
   source fashion-rag/bin/activate
   ```
2. Upgrade pip and install dependencies
   ```sh
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. Create a .env file like the following:
   ```js
   export WANDB_API_KEY="ENTER YOUR WANDB TOKEN"
   export TORCH_HOME="ENTER YOUR TORCH PATH TO SAVE TORCH MODELS USED FOR METRICS COMPUTING"
   export HF_TOKEN="ENTER YOUR HUGGINGFACE TOKEN"
   export HF_CACHE_DIR="PATH WHERE YOU WANT TO SAVE THE HF MODELS (NEED CUSTOM VARIABLE TO ACCOUNT FOR OLD TRANSFORMERS PACKAGES, OTHERWISE USE HF_HOME)"
   ```

<!-- USAGE EXAMPLES -->

## Data and Models
Download DressCode from the [original repository](https://github.com/aimagelab/dress-code)
Download the finetuned U-Net and Inversion Adapter from [this source](https://huggingface.co/furio19/fashion-rag) and put them into your experiment folder as follows:
```plaintext
<experiment folder>/
│
├── unet_120000.pth
├── inversion_adapter_120000.pth
```


## Inference
Let's generate our virtual dressing images using the finetuned TEMU-VTOFF model.
```sh
source fashion-rag/bin/activate

python evaluate_RAG.py \
    python evaluate_RAG.py \
    --pretrained_model_name_or_path stabilityai/stable-diffusion-2-inpainting \
    --output_dir "output directory path" \
    --finetuned_models_dir "U-Net and inversion adapter directory weights path" \
    --unet_name unet_120000.pth --inversion_adapter_name inversion_adapter_120000.pth \
    --dataset dresscode --dresscode_dataroot <data path>/DressCode \
    --category "garment category"\
    --test_order "specify paired or unpaired" --mask_type mask \
    --phase test --num_inference_steps 50 \
    --test_batch_size 8 --num_workers_test 8 --metrics_batch_size 8 --mixed_precision fp16 \
    --text_usage prompt_noun_chunks \
    --retrieve_path <data path>/DressCode/fashion-rag-retrieval \
    --clip_retrieve_model ViT-L-14 --clip_retrieve_weights laion2b_s32b_b82k \
    --n_chunks "number of text chunks 1 to 3" \
    --n_retrieved "number of retrieved images 1 to 3" \
    --metrics fid_score kid_score retrieved_score clip_score lpips_score ssim_score \
    --attention_layers_fine_list '-1' '0 1 2 3'\
    --compute_metrics
```
```plaintext
out_dir/pte_paired_nc_<number_of_chunks>_nr_<number_of_retrieved_images>/
│
├── lower_body/
├── upper_body/
├── dresses/
└── metrics_all.json
```