import argparse
import gc
import json
import os
from pathlib import Path
from typing import List
import sys

PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import PIL.Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms
from utils.cleanfid_v0_1_35 import fid # cannot use package fid here and custom fid in make_custom_stats
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision import transforms
from torchvision.ops import masks_to_boxes
from tqdm import tqdm

from dataset.dresscode import DressCodeDataset, DressCodeRetrieval, retrieve_collate_fn
from dataset.vitonhd import VitonHDDataset
from utils.generated_fid_stats import make_custom_stats
from models.sketcher import Sketcher

PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()


class GenTestDataset(torch.utils.data.Dataset):
    def __init__(self, gen_folder, category, transform):
        assert category in ['all', 'dresses', 'lower_body', 'upper_body'], 'Unsupported category'

        self.category = category
        self.transform = transform
        self.gen_folder = gen_folder

        if category in ['lower_body', 'upper_body', 'dresses']:
            # Sort by name + condition name (delete .split("_"[0]), int())
            self.paths = sorted(
                [os.path.join(gen_folder, category, name) for name in os.listdir(os.path.join(gen_folder, category))],
                key=lambda x: os.path.splitext(os.path.basename(x))[0])
        elif category == 'all':
            existing_categories = []
            for category in ['lower_body', 'upper_body', 'dresses']:
                if os.path.exists(os.path.join(gen_folder, category)):
                    existing_categories.append(category)

            self.paths = sorted(
                [os.path.join(gen_folder, category, name) for category in existing_categories for
                 name in os.listdir(os.path.join(gen_folder, category)) if
                 os.path.exists(os.path.join(gen_folder, category, name))],
                key=lambda x: os.path.splitext(os.path.basename(x))[0])
        else:
            raise ValueError('Unsupported category')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        name = os.path.splitext(os.path.basename(path))[0]
        img = self.transform(PIL.Image.open(path).convert('RGB'))
        return img, name


@torch.inference_mode()
def sketch_metric(gt_sketch_batch: torch.tensor, gen_batch: torch.tensor, generated_masks: torch.tensor,
                  sketcher: Sketcher, criterion: callable, ):
    """Computes the sketch metric.
    Args:
        im_cloth_batch: Batch of cloth images
        gen_batch: Batch of generated images
        sketcher: The sketcher model
        criterion: The criterion to use
        labelmap_generator: The labelmap generator
        categories: The categories to compute the metric for
    Returns:
        The metric
    """

    scaled_gen_batch = (gen_batch - 0.5) * 2  # scale to [-1, 1]

    # get mask
    gen_masked_images = gen_batch * generated_masks
    gen_masked_images[gen_masked_images == 0] = 1  # White background works better

    # normalize imagenet mean and std
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    gen_sketch = sketcher(normalize(gen_masked_images))

    gt_black = torch.sum(gt_sketch_batch > 0.01, dim=(1, 2, 3))
    return torch.sum(criterion(gen_sketch, gt_sketch_batch).sum((1, 2, 3)) / gt_black)


@torch.inference_mode()
def compute_metrics(gen_folder, setting, dataset, category: str, metrics2compute: List[str], dresscode_dataroot,
                    vitonhd_dataroot, generated_size=(512, 384), batch_size=32, workers=8, 
                    texture_order='original', use_chunks=False, n_chunks=1, n_retrieved=1, aug_captions=False,
                    retrieve_path = None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    assert setting in ['paired', 'unpaired']
    assert dataset in ['dresscode', 'vitonhd'], 'Unsupported dataset'
    assert category in ['all', 'dresses', 'lower_body', 'upper_body'], 'Unsupported category'
    assert texture_order in ['original', 'shuffled']

    if dataset == 'dresscode':
        gt_folder = dresscode_dataroot
    elif dataset == 'vitonhd':
        gt_folder = vitonhd_dataroot
    else:
        raise ValueError('Unsupported dataset')

    for m in metrics2compute:
        assert m in ['all', 'ssim_score', 'lpips_score', 'fid_score', 'kid_score', 'is_score',
                     'clip_score', 'pose_score', 'sketch_score', 'retrieved_score'], 'Unsupported metric'

    if metrics2compute == ['all']:
        metrics2compute = ['ssim_score', 'lpips_score', 'fid_score', 'kid_score', 'is_score', 'clip_score',
                           'pose_score', 'sketch_score', 'retrieved_score']

    if n_retrieved==0 and 'retrieved_score' in metrics2compute: 
        print("you provided CLIP-I without conditioning images, dropping this metric")
        metrics2compute.remove('retrieved_score')

    # instantiate needed models
    print("Loading custom stats and computing Fid/Kid scores")
    if category == 'all':
        if "fid_score" in metrics2compute or "all" in metrics2compute:
            print("Computing FID Score for all categories")
            if not fid.test_stats_exists(f"{dataset}_all", mode='clean'):
                make_custom_stats(dresscode_dataroot, vitonhd_dataroot)
            fid_score = fid.compute_fid(gen_folder, dataset_name=f"{dataset}_all", mode='clean', dataset_split="custom",
                                        verbose=True, use_dataparallel=False)
            print(f"FID score for {dataset}_all: {fid_score}")
        if "kid_score" in metrics2compute or "all" in metrics2compute:
            print("Computing KID Score for all categories")
            if not fid.test_stats_exists(f"{dataset}_all", mode='clean'):
                make_custom_stats(dresscode_dataroot, vitonhd_dataroot)
            kid_score = fid.compute_kid(gen_folder, dataset_name=f"{dataset}_all", mode='clean', dataset_split="custom",
                                        verbose=True, use_dataparallel=False)
            print(f"KID score for {dataset}_all: {kid_score}")
    else:
        if "fid_score" in metrics2compute or "all" in metrics2compute:
            print("Computing FID Score for category:", category)
            if not fid.test_stats_exists(f"{dataset}_{category}", mode='clean'):
                make_custom_stats(dresscode_dataroot, vitonhd_dataroot)
            fid_score = fid.compute_fid(os.path.join(gen_folder, category), dataset_name=f"{dataset}_{category}",
                                        mode='clean', verbose=True, dataset_split="custom", use_dataparallel=False)
            print(f"FID score for {dataset}_{category}: {fid_score}")
        if "kid_score" in metrics2compute or "all" in metrics2compute:
            print("Computing KID Score for category:", category)
            if not fid.test_stats_exists(f"{dataset}_{category}", mode='clean'):
                make_custom_stats(dresscode_dataroot, vitonhd_dataroot)
            kid_score = fid.compute_kid(os.path.join(gen_folder, category),
                                        dataset_name=f"{dataset}_{category}", mode='clean', verbose=True,
                                        dataset_split="custom", use_dataparallel=False)
            print(f"KID score for {dataset}_{category}: {kid_score}")

    if "pose_score" in metrics2compute or "all" in metrics2compute:
        from utils.pose_metric import compute_metrics_category_wise
        pose_score = compute_metrics_category_wise(gen_folder, dresscode_dataroot, vitonhd_dataroot, dataset, category,
                                                   setting, generated_size)

    trans = transforms.Compose([
        transforms.Resize(generated_size),
        transforms.ToTensor(),
    ])

    gen_dataset = GenTestDataset(gen_folder, category, transform=trans)
    # gt_dataset = GTTestDataset(gt_folder, dataset, category, trans)

    outputlist = ['image', 'im_name', 'c_name', 'captions', 'inpaint_mask', 'category', 'cloth']
    retrievelist=[]
    if 'sketch_score' in metrics2compute or 'all' in metrics2compute:
        outputlist.append('greyscale_im_sketch')

    if 'retrieved_score' in metrics2compute or 'all' in metrics2compute:
        retrievelist.append('retrieved_cloth')
        retrieved_score = 0

    if dataset == 'vitonhd':
        gt_dataset = VitonHDDataset(vitonhd_dataroot, 'test', order=setting, outputlist=outputlist, size=generated_size,
                                    texture_order=texture_order)
    elif dataset == 'dresscode':
        if category == 'all':
            gt_dataset = DressCodeRetrieval(dresscode_dataroot, phase='test', order=setting, outputlist=outputlist, retrievelist=retrievelist,
                                          size=generated_size, texture_order=texture_order, n_chunks=n_chunks, top_k=n_retrieved, retrieve_feat_path=retrieve_path,
                                          augment_dataset=aug_captions)
        else:
            gt_dataset = DressCodeRetrieval(dresscode_dataroot, phase='test', order=setting, outputlist=outputlist, retrievelist=retrievelist,
                                          size=generated_size, category=[category], texture_order=texture_order,
                                          n_chunks=n_chunks, top_k=n_retrieved, retrieve_feat_path=retrieve_path,
                                          augment_dataset=aug_captions)
    else:
        raise ValueError('Unsupported dataset')

    # Make the order of the images in the datasets the same
    # CAREFUL: 
    # sorting_indices = np.argsort(gt_dataset.im_names) # original
    gen_names = [os.path.splitext(os.path.basename(path))[0] for path in gen_dataset.paths]
    # sorted(sorting_names, key=lambda x: x.split("_")[0] + "_".join(x.split("_")[2:]))[0]
    # HP Respect the sorting of genDataset (done inside GenDataset)
    if hasattr(gt_dataset, "captions"):
        sorting_names = [im_name.split('.')[0] + "_" + "_".join(["_".join(chunk.split(" ")) for chunk in caption.split(", ")]).replace("/", "_") for im_name, caption in zip(gt_dataset.im_names, gt_dataset.captions)]
    else:
        sorting_names = [im_name.split('.')[0] for im_name in gt_dataset.im_names]
    sorting_indices = np.argsort(sorting_names) # assumes 1 image generated per sample + len(gen_dataset == len(gt_dataset))
    # assumes that getitem has access to im_names, c_names, dataroot_names, captions
    # gen_dataset.paths =  np.array(gen_dataset.paths)[sorting_indices].tolist()
    gt_dataset.im_names = np.array(gt_dataset.im_names)[sorting_indices].tolist()
    gt_dataset.c_names = np.array(gt_dataset.c_names)[sorting_indices].tolist()
    gt_dataset.dataroot_names = np.array(gt_dataset.dataroot_names)[sorting_indices].tolist()
    if hasattr(gt_dataset, "captions"):
        gt_dataset.captions = np.array(gt_dataset.captions)[sorting_indices].tolist() 
        captions = gt_dataset.captions

    # we expect gen_name = filenum_{0 or 1}_{caption or c_name}
    for gen_name, im_name, c_name, caption in zip(gen_names, gt_dataset.im_names, gt_dataset.c_names, captions):
        assert gen_name.split('_')[0] == im_name.split('_')[0], print(f"gen: {gen_name.split('_')[0]}, im: {im_name.split('_')[0]}")
        if not use_chunks: 
            assert gen_name.split('_')[-2] == c_name.split('_')[0]
        else:
            save_caption = "_".join(["_".join(chunk.split(" ")) for chunk in caption.split(", ")]).replace("/", "_") # n_nounchunks fisso
            assert '_'.join(gen_name.split('_')[2:]) == save_caption, print("second name not equal", gen_name, save_caption)



    gen_loader = DataLoader(gen_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    from functools import partial
    collate_fn = partial(retrieve_collate_fn, var_keys=retrievelist, n_retrieved=n_retrieved)
    gt_loader = DataLoader(gt_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn = collate_fn)

    if "is_score" in metrics2compute or "all" in metrics2compute:
        model_is = InceptionScore(normalize=True).to(device)

    if "ssim_score" in metrics2compute or "all" in metrics2compute:
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    if "lpips_score" in metrics2compute or "all" in metrics2compute:
        lpips = LearnedPerceptualImagePatchSimilarity(net='alex', normalize=True).to(device)

    if 'clip_score' in metrics2compute or 'all' in metrics2compute or 'retrieved_score' in metrics2compute:
        from torchmetrics.multimodal import CLIPScore
        model_clipscore = CLIPScore(model_name_or_path="laion/CLIP-ViT-H-14-laion2B-s32B-b79K").to(device) # "openai/clip-vit-base-patch16" "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

    for idx, (gen_batch, gt_batch) in tqdm(enumerate(zip(gen_loader, gt_loader)), total=len(gt_loader)):
        # if idx*batch_size > 100 and debug: break
        gen_images, gen_names = gen_batch
        gen_images = gen_images.to(device)

        gt_images = (gt_batch['image'] + 1) / 2
        gt_images = gt_images.to(device)

        gt_greyscale_sketch = gt_batch.get('greyscale_im_sketch')
        if gt_greyscale_sketch is not None:
            gt_greyscale_sketch = gt_greyscale_sketch.to(device)  # [0,1]
        gt_texture = gt_batch.get('texture')
        if gt_texture is not None:
            gt_texture = (gt_texture + 1) / 2
            gt_texture = gt_texture.to(device)

        gt_im_names = gt_batch['im_name']
        gt_c_names = gt_batch['c_name']
        gt_captions = gt_batch['captions']
        gt_masks = gt_batch['inpaint_mask']
        gt_category = gt_batch['category']
        captions = gt_batch['captions']
        retrieved_cloths = gt_batch.get("retrieved_cloth")
        if retrieved_cloths is not None:
            retrieved_cloths = (retrieved_cloths + 1) / 2

        # not needed anymore
        for gen_name, im_name, c_name, caption in zip(gen_names, gt_im_names, gt_c_names, captions):
            # check matching names here
            assert gen_name.split('_')[0] == im_name.split('_')[0], print(f"gen: {gen_name.split('_')[0]}, im: {im_name.split('_')[0]}")
            if not use_chunks: 
                assert gen_name.split('_')[-2] == c_name.split('_')[0]
            else:
                save_caption = "_".join(["_".join(chunk.split(" ")) for chunk in caption.split(", ")]).replace("/", "_")
                assert '_'.join(gen_name.split('_')[2:]) == save_caption, print("second name not equal")

        if "is_score" in metrics2compute or "all" in metrics2compute:
            model_is.update(gen_images)

        if "ssim_score" in metrics2compute or "all" in metrics2compute:
            if setting == 'paired':
                ssim.update(gen_images, gt_images.to(device))
            else:
                pass

        if "lpips_score" in metrics2compute or "all" in metrics2compute:
            if setting == 'paired':
                lpips.update(gen_images, gt_images.to(device))
            else:
                pass

        if "clip_score" in metrics2compute or "all" in metrics2compute:
            # compute clipscore
            bboxes = masks_to_boxes(gt_masks[:, 0, ...])
            bboxes = bboxes.type(torch.int32)  # xmin, ymin, xmax, ymax format
            padded_imgs = []
            for i in range(len(bboxes)):
                xmin = bboxes[i, 0]
                xmax = bboxes[i, 2]
                ymin = bboxes[i, 1]
                ymax = bboxes[i, 3]

                cropped_img = gen_images[i, :, ymin:ymax + 1, xmin:xmax + 1].clone()

                # always use mask to delete background
                cropped_mask = gt_masks[i, 0, ymin:ymax + 1, xmin:xmax + 1].clone()
                cropped_img[:, cropped_mask == 0] = 1

                _, h, w = cropped_img.shape

                if h > w:
                    h_new = 224
                    w_new = round(h_new * w / h)
                    padded_img = transforms.Resize((h_new, w_new), antialias=True)(cropped_img)
                    # left and right padding only
                    left_pad = round((224 - w_new) / 2)
                    right_pad = 224 - w_new - left_pad
                    padded_img = transforms.Pad([left_pad, 0, right_pad, 0], fill=1)(padded_img)
                else:
                    w_new = 224
                    h_new = round(w_new * h / w)
                    padded_img = transforms.Resize((h_new, w_new), antialias=True)(cropped_img)
                    # top and bottom padding only
                    top_pad = round((224 - h_new) / 2)
                    bottom_pad = 224 - h_new - top_pad
                    padded_img = transforms.Pad([0, top_pad, 0, bottom_pad], fill=1)(padded_img)

                padded_imgs.append(padded_img)

            concat_padded_imgs = torch.stack(padded_imgs)
            model_clipscore.update(concat_padded_imgs.clamp(0, 1), gt_captions)

        if ("retrieved_score" in metrics2compute or "all" in metrics2compute) and n_retrieved>0:
            bboxes = masks_to_boxes(gt_masks[:, 0, ...]) # warped mask
            bboxes = bboxes.type(torch.int32)  # xmin, ymin, xmax, ymax format
            cropped_imgs = []
            for i in range(len(bboxes)):  # batch
                xmin = bboxes[i, 0]
                xmax = bboxes[i, 2]
                ymin = bboxes[i, 1]
                ymax = bboxes[i, 3]

                cropped_img = gen_images[i, :, ymin:ymax + 1, xmin:xmax + 1].clone()

                # always use mask to delete background
                cropped_mask = gt_masks[i, 0, ymin:ymax + 1, xmin:xmax + 1].clone()
                cropped_img[:, cropped_mask == 0] = 1 # should I do it on retrieved too?

                if gt_category[i] == 'lower_body':
                    cropped_imgs.append(
                        torchvision.transforms.functional.crop(cropped_img, 15, (xmax - xmin) // 2 - 32, 64, 64))
                else:
                    cropped_imgs.append(torchvision.transforms.CenterCrop(64)(cropped_img))

            bs = len(cropped_imgs) # needed for multiple images reshaping
            # concat_cropped_imgs = torch.stack(cropped_imgs)

            processed_images = model_clipscore.processor(images=list(cropped_imgs), return_tensors="pt")
            cloth_features = model_clipscore.model.get_image_features(processed_images["pixel_values"].to(device))
            cloth_features = F.normalize(cloth_features)

            processed_cloth = model_clipscore.processor(images=list(retrieved_cloths), return_tensors="pt")
            retrieved_cloth_features = model_clipscore.model.get_image_features(processed_cloth["pixel_values"].to(device))
            ensemble_feats = F.normalize(retrieved_cloth_features).view(bs,-1, *retrieved_cloth_features.shape[1:])
            avg_scores = [torch.cosine_similarity(cloth_features, ensemble_feats[:,i, ...]).sum().item() for i in range(ensemble_feats.shape[1])]

            retrieved_score += sum(avg_scores)/len(avg_scores) # score for the full enseble of images

        gc.collect()
        torch.cuda.empty_cache()

    if "is_score" in metrics2compute or "all" in metrics2compute:
        is_score, is_std = model_is.compute()
    if "ssim_score" in metrics2compute or "all" in metrics2compute:
        if setting == 'paired':
            ssim_score = ssim.compute()
        else:
            ssim_score = 0
    if "lpips_score" in metrics2compute or "all" in metrics2compute:
        if setting == 'paired':
            lpips_score = lpips.compute()
        else:
            lpips_score = 0

    if "retrieved_score" in metrics2compute or "all" in metrics2compute:
        retrieved_score = retrieved_score / len(gen_dataset)

    if "clip_score" in metrics2compute or "all" in metrics2compute:
        clip_score = model_clipscore.compute()

    results = {}

    for m in metrics2compute:
        if torch.is_tensor(locals()[m]):
            results[m] = locals()[m].item()
        else:
            results[m] = locals()[m]

    if "clip_score" in metrics2compute or "all" in metrics2compute or "retrieved_score" in metrics2compute:
        del model_clipscore
    if "is_score" in metrics2compute or "all" in metrics2compute:
        del model_is
    if "ssim_score" in metrics2compute or "all" in metrics2compute:
        del ssim
    if "lpips_score" in metrics2compute or "all" in metrics2compute:
        del lpips

    gc.collect()
    torch.cuda.empty_cache()
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_folder', type=str, default='/equilibrium/abaldrati/datasets/DressCode')
    parser.add_argument('--gen_folder', type=str,
                        default='/equilibrium/abaldrati/pami_mgd/checkpoints/texture_sketch_nounchunks_invFT_vstar16/200k_scr0.2_alf32_unpaired')
    parser.add_argument('--setting', type=str, default='unpaired')
    parser.add_argument('--dataset', type=str, default='dresscode')
    parser.add_argument('--category', type=str, default='all')
    parser.add_argument('--metrics2compute', type=str, default='lpips_score')
    parser.add_argument('--dresscode_dataroot', type=str, default='/equilibrium/abaldrati/datasets/DressCode')
    parser.add_argument('--vitonhd_dataroot', type=str, default='/andromeda/personal/abaldrati/datasets/viton_hd')
    parser.add_argument('--save_dir_metrics', type=str, default='/work/CucchiaraYOOX2019/diffusion_vto/metrics')
    parser.add_argument('--test_name', type=str, default='VITON-HD_paired_1024x768')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_worksers", type=int, default=20)
    parser.add_argument('--retrieve_feat_path', type=str, default="/work/CucchiaraYOOX2019/dataset/DressCode/features/ViT-L-14_laion2b_s32b_b82k/metrics")
    args = parser.parse_args()

    print(torch.cuda.is_available())
    pred_files_paths = []
    chunk_combos = ["single", "double", "triplet"]
    for combo in chunk_combos:
        for cat in ["dresses", "lower_body", "upper_body"]:
            pred_files_paths.append(os.path.join(args.retrieve_feat_path, "metrics", f"test_{cat}_{combo}.json"))

    results = compute_metrics(args.gen_folder, args.setting, args.dataset, args.category,
                              ["retrieved_score"],
                              args.dresscode_dataroot, args.vitonhd_dataroot, batch_size=args.batch_size,
                              workers=args.num_worksers, use_chunks=True, n_chunks=3, n_retrieved=3, pred_files_paths=pred_files_paths)

    for k, v in results.items():
        if torch.is_tensor(v):
            results[k] = v.item()
        else:
            results[k] = v
        print(f'{k}: {v}')

    # # save results
    # if not os.path.exists(args.save_dir_metrics):
    #     os.makedirs(args.save_dir_metrics, exist_ok=True)
    #
    # with open(os.path.join(args.save_dir_metrics, f'{args.test_name}.json'), 'w') as f:
    #     json.dump(results, f)
    #
    # print(results)
