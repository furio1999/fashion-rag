from pathlib import Path
from typing import List, Tuple
import PIL.Image
import torch
import os
import json

import torchvision
from tqdm import tqdm
import openpifpaf
import glob

PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()

import pandas as pd


# def get_categories(dataroot_path: str, order: str, categories: List[str]) -> dict:
#     for c in categories:
#         dataroot = os.path.join(dataroot_path, c)
#         filename = os.path.join(dataroot, f"test_pairs_{order}.txt")
#
#         cats = {}
#         with open(filename, 'r') as f:
#             for line in f.readlines():
#                 im_name, _ = line.strip().split()
#                 cat = dataroot.split('/')[-1]
#                 cats[im_name.split("/")[0]] = cat
#
#     return cats


def get_element_by_category(dataroot_path: str, dataset, order: str, category: str) -> dict:
    if dataset == 'dresscode':
        dataroot = os.path.join(dataroot_path, category)
        filename = os.path.join(dataroot, f"test_pairs_{order}.txt")
    else:  # vitonhd
        filename = os.path.join(dataroot_path, f"test_pairs.txt")

    image_list = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            im_name, _ = line.strip().split()
            image_list.append(im_name.split("/")[0])
    return image_list


@torch.inference_mode()
def extract_keypoints_dict_by_category(path: str, image_list, category: str, dim: Tuple[int, int]) -> dict:
    print(f"Extracting {category} keypoints...")
    image_list_noext = [os.path.splitext(img)[0] for img in image_list]

    file_list = [p for p in glob.glob(os.path.join(path, f'*_0*')) if  # _0 filter out the in-shop cloth images
                 (os.path.splitext(os.path.basename(p))[0] in image_list_noext or
                  "_".join(os.path.splitext(os.path.basename(p))[0].split("_")[:2]) in image_list_noext)]
    # We need to manage the different naming of the generated images

    assert len(file_list) > 0

    predictions = {}

    predictor = openpifpaf.Predictor(checkpoint='tshufflenetv2k30')
    # predictor.long_edge = dim[0]
    # predictor.preprocess = predictor._preprocess_factory()
    # predictor.preprocess.preprocess_list.insert(0, torchvision.transforms.Resize(dim))
    # predictor.preprocess.preprocess_list[1] = openpifpaf.transforms.RescaleAbsolute(dim[0])
    for path in tqdm(file_list):
        pil_image = PIL.Image.open(path).resize((dim[1], dim[0]))
        pred = predictor.pil_image(pil_image)[0]
        try:
            predictions[path.split("/")[-1].split("_")[0]] = {k: v.tolist() for k, v in
                                                              zip(pred[0].keypoints[5:], pred[0].data[5:])}
        except Exception as e:
            print(f"no keypoints found for {path.split('/')[-1]}")
            print(e)


    # for (pred, _, meta), name in tqdm(zip(predictor.images(file_list), file_list), total=len(file_list)):
    #     try:
    #         predictions[name.split("/")[-1].split("_")[0]] = {k: v.tolist() for k, v in
    #                                                           zip(pred[0].keypoints[5:], pred[0].data[5:])}
    #     except Exception as e:
    #         print(f"no keypoints found for {name.split('/')[-1]}")
    #         print(e)

    return predictions


@torch.inference_mode()
def compute_metrics_category_wise(gen_folder, dresscode_dataroot, vitonhd_dataroot, dataset: str, category, setting,
                                  generated_size):
    total_detections = torch.zeros(3)
    gt_keypoints_dict = {}
    generated_keypoints_dict = {}

    if category == 'all' and dataset == 'dresscode':
        category = ('dresses', 'lower_body', 'upper_body')
    elif category == 'all' and dataset == 'vitonhd':
        category = ('upper_body')
    else:
        category = tuple([category])

    if dataset == 'dresscode':
        dataset_dataroot = dresscode_dataroot
    else:  # vitonhd
        dataset_dataroot = vitonhd_dataroot

    for id, c in enumerate(category):
        # print(f"Processing category {c}...")

        category_images_list = get_element_by_category(dataset_dataroot, dataset, setting, c)

        gt_keypoints = []
        generated_keypoints = []
        category_filters = []

        if generated_size == (256, 192):
            keypoint_gt_path_root = os.path.join(PROJECT_ROOT, 'data', 'keypoints', dataset, 'original_imgs_256x192')
        else:
            keypoint_gt_path_root = os.path.join(PROJECT_ROOT, 'data', 'keypoints', dataset, 'original_imgs')
        try:
            with open(os.path.join(keypoint_gt_path_root, f'keypoints_gt_{c}.json')) as f:
                gt_keypoints_dict[c] = json.load(f)
        except FileNotFoundError:
            if dataset == 'dresscode':
                gt_keypoints_dict[c] = extract_keypoints_dict_by_category(
                    os.path.join(dataset_dataroot, c, 'images'), category_images_list, c, generated_size)
            else:  # vitonhd
                gt_keypoints_dict[c] = extract_keypoints_dict_by_category(
                    os.path.join(dataset_dataroot, 'test', 'image'), category_images_list, c, generated_size)

            os.makedirs(os.path.join(keypoint_gt_path_root), exist_ok=True)
            with open(os.path.join(keypoint_gt_path_root, f"keypoints_gt_{c}.json"), "w") as f:
                json.dump(gt_keypoints_dict[c], f)

        try:
            with open(os.path.join(gen_folder, f"keypoints_generated_{c}.json")) as f:
                generated_keypoints_dict[c] = json.load(f)
        except:
            generated_keypoints_dict[c] = extract_keypoints_dict_by_category(
                os.path.join(gen_folder, c), category_images_list, c, generated_size)

            with open(os.path.join(gen_folder, f"keypoints_generated_{c}.json"), "w") as f:
                json.dump(generated_keypoints_dict[c], f)

        for k, v in tqdm(gt_keypoints_dict[c].items()):
            try:
                gn_kp = generated_keypoints_dict[c][k]
                total_detections[id] += 1
            except:
                print(f"no keypoints found for {k}")
                continue

            if c == 'dresses':
                category_filter = torch.ones(12)
            elif c == 'upper_body':
                category_filter = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
            elif c == 'lower_body':
                category_filter = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
            else:
                raise ValueError("Wrong category")

            gt_keypoints.append(list(v.values()))
            generated_keypoints.append(list(gn_kp.values()))
            category_filters.append(category_filter)

        if c == 'dresses':
            dress_category_filters = torch.stack(category_filters)
            dress_gt_keypoints = torch.tensor(gt_keypoints)[:, :, :2]
            dress_confidence = torch.tensor(gt_keypoints)[:, :, 2]
            dress_confidence_gen = torch.tensor(generated_keypoints)[:, :, 2]
            dress_generated_keypoints = torch.tensor(generated_keypoints)[:, :, :2]
        elif c == 'upper_body':
            upper_body_category_filters = torch.stack(category_filters)
            upper_body_gt_keypoints = torch.tensor(gt_keypoints)[:, :, :2]
            upper_body_confidence = torch.tensor(gt_keypoints)[:, :, 2]
            upper_body_confidence_gen = torch.tensor(generated_keypoints)[:, :, 2]
            upper_body_generated_keypoints = torch.tensor(generated_keypoints)[:, :, :2]
        elif c == 'lower_body':
            lower_body_body_category_filters = torch.stack(category_filters)
            lower_body_gt_keypoints = torch.tensor(gt_keypoints)[:, :, :2]
            lower_body_confidence = torch.tensor(gt_keypoints)[:, :, 2]
            lower_body_confidence_gen = torch.tensor(generated_keypoints)[:, :, 2]
            lower_body_generated_keypoints = torch.tensor(generated_keypoints)[:, :, :2]
        else:
            raise ValueError("Wrong category")

    results = {}
    results["summary"] = {}

    keypoint_wise_score = []
    if category[0] == 'upper_body' or len(category) == 3:
        upper_body_diff = (
                (upper_body_gt_keypoints - upper_body_generated_keypoints).pow(2).sum(-1).sqrt() * torch.round(
            upper_body_confidence) * torch.round(upper_body_confidence_gen) * upper_body_category_filters)
        upper_num_kp = (torch.round(upper_body_confidence) * torch.round(
            upper_body_confidence_gen) * upper_body_category_filters).sum()
        results["summary"]["upper_num_kp"] = upper_num_kp.item()
        upper_body_score = upper_body_diff.sum() / upper_num_kp
        results["summary"]["upper_body"] = upper_body_score.item()
        keypoint_wise_score.append(upper_body_diff.sum(0) / (torch.round(upper_body_confidence) * torch.round(
            upper_body_confidence_gen) * upper_body_category_filters).sum(0))
    if category[0] == 'lower_body' or len(category) == 3:
        lower_body_diff = (
                (lower_body_gt_keypoints - lower_body_generated_keypoints).pow(2).sum(-1).sqrt() * torch.round(
            lower_body_confidence) * torch.round(lower_body_confidence_gen) * lower_body_body_category_filters)
        lower_num_kp = (torch.round(lower_body_confidence) * torch.round(
            lower_body_confidence_gen) * lower_body_body_category_filters).sum()
        results["summary"]["lower_num_kp"] = lower_num_kp.item()
        lower_body_score = lower_body_diff.sum() / lower_num_kp
        results["summary"]["lower_body"] = lower_body_score.item()
        keypoint_wise_score.append(lower_body_diff.sum(0) / (torch.round(lower_body_confidence) * torch.round(
            lower_body_confidence_gen) * lower_body_body_category_filters).sum(0))
    if category[0] == 'dresses' or len(category) == 3:
        dress_diff = ((dress_gt_keypoints - dress_generated_keypoints).pow(2).sum(-1).sqrt() * torch.round(
            dress_confidence) * torch.round(dress_confidence_gen) * dress_category_filters)
        dress_num_kp = (
                torch.round(dress_confidence) * torch.round(dress_confidence_gen) * dress_category_filters).sum()
        results["summary"]["dress_num_kp"] = dress_num_kp.item()
        dress_score = dress_diff.sum() / dress_num_kp
        results["summary"]["dresses"] = dress_score.item()
        keypoint_wise_score.append(dress_diff.sum(0) / (
                torch.round(dress_confidence) * torch.round(dress_confidence_gen) * dress_category_filters).sum(0))

    if len(category) == 3:
        # mean over all keypoints of all images over detections found in gt
        all_score = (dress_diff.sum() + upper_body_diff.sum() + lower_body_diff.sum()) / (
                dress_num_kp + upper_num_kp + lower_num_kp)
        print(f"{'-' * 40}")
        print(f"Overall mean keypoint error: {all_score}")
        results["summary"]["all"] = all_score.item()

    # category score

    # print(f"{'-'*40}")
    # print(f"Overall category mean keypoint error")
    # print()
    # print(f"Upper body: {upper_body_score}")
    # print(f"Lower body: {lower_body_score}")
    # print(f"Dresses: {dress_score}")

    # mean over each keypoint for all images

    print(f"{'-' * 40}")
    print(f"Mean error for each keypoint:")
    results["summary"]["total_detections"] = {}

    for id, c in enumerate(category):
        results["summary"]["total_detections"][c] = total_detections[id].item()
        results[c] = {}
        print()
        print(f"{c.upper()}:")
        print()
        for k, v in zip(gt_keypoints_dict[c][list(gt_keypoints_dict[c].keys())[0]].keys(), keypoint_wise_score[id]):
            if v > 0:
                print(f"{k}: {v}")
                results[c][k] = v.item()

    return results
