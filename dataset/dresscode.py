import glob
import json
import os
import random
from pathlib import Path
from typing import Tuple, Literal

import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageOps
from torchvision.ops import masks_to_boxes
from torch.utils.data._utils.collate import default_collate
import torch.nn.functional as F

from utils.labelmap import label_map
from utils.posemap import kpoint_to_heatmap
from itertools import combinations
from collections import OrderedDict
import random

torch.multiprocessing.set_sharing_strategy('file_system')


class DressCodeDataset(data.Dataset):
    """
    Dataset class for the Dress Code Multimodal Dataset
    im_parse: inpaint mask
    """

    def __init__(self,
                 dataroot_path: str,
                 phase: Literal["train", "test"],
                 radius: int = 5,
                 caption_file: str = 'fine_captions.json',
                 coarse_caption_file: str = 'coarse_captions.json',
                 sketch_threshold_range: Tuple[int, int] = (20, 127),
                 order: Literal['paired', 'unpaired'] = 'paired',
                 outputlist: Tuple[str] = ('c_name', 'im_name', 'image', 'im_cloth', 'cloth', 'shape', 'pose_map',
                                           'parse_array', 'im_mask', 'inpaint_mask', 'parse_mask_total',
                                           'im_sketch', 'greyscale_im_sketch', 'captions', 'category', 'stitch_label',
                                           'texture'),
                 category: Literal['dresses', 'upper_body', 'lower_body'] = ('dresses', 'upper_body', 'lower_body'),
                 size: Tuple[int, int] = (512, 384),
                 mask_type: Literal["mask", "bounding_box"] = "bounding_box",
                 texture_order: Literal["shuffled", "original"] = "original",
                 ):

        super().__init__()
        self.dataroot = dataroot_path
        self.phase = phase
        self.sketch_threshold_range = sketch_threshold_range
        self.category = category
        self.outputlist = outputlist
        self.height = size[0]
        self.width = size[1]
        self.radius = radius
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transform2d = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.order = order
        self.mask_type = mask_type
        self.texture_order = texture_order

        im_names = []
        c_names = []
        dataroot_names = []
        category_names = [] # NEW

        possible_outputs = ['c_name', 'im_name', 'cpath', 'cloth', 'image', 'im_cloth', 'shape', 'im_head', 'im_pose',
                            'pose_map', 'parse_array', 'dense_labels', 'dense_uv', 'skeleton', 'im_mask',
                            'inpaint_mask', 'greyscale_im_sketch', 'parse_mask_total', 'cloth_sketch', 'im_sketch',
                            'captions', 'category', 'hands', 'parse_head_2', 'stitch_label', 'texture', 'parse_cloth'] # if x in output, then MUST be initializated as a variable

        assert all(x in possible_outputs for x in outputlist)

        # Load Captions
        with open(os.path.join(self.dataroot, caption_file)) as f:
            self.captions_dict = json.load(f)
        self.captions_dict = {k: v for k, v in self.captions_dict.items() if len(v) >= 3}
        with open(os.path.join(self.dataroot, coarse_caption_file)) as f:
            self.captions_dict.update(json.load(f))

        # annotated_elements = [k for k, _ in self.captions_dict.items()]
        texture_mapping = {}  # {model_name: texture_name} used only when texture_order == "shuffled"
        for c in sorted(category):
            assert c in ['dresses', 'upper_body', 'lower_body']

            dataroot = os.path.join(self.dataroot, c)
            if phase == 'train':
                filename = os.path.join(dataroot, f"{phase}_pairs.txt")
            else:
                filename = os.path.join(dataroot, f"{phase}_pairs_{order}.txt")

            texture_filename = os.path.join(dataroot, f"test_shuffled_textures.txt")

            with open(filename, 'r') as f:
                for line in f.readlines():
                    im_name, c_name = line.strip().split()
                    if c_name.split('_')[0] not in self.captions_dict:
                        continue
                    # avoid duplicates
                    if im_name in im_names:
                        continue

                    im_names.append(im_name)
                    c_names.append(c_name)
                    dataroot_names.append(dataroot)
                    category_names.append(c) # NEW

            if texture_order == "shuffled":
                with open(texture_filename, 'r') as f:
                    for line in f.readlines():
                        im_name, texture_name = line.strip().split()
                        texture_mapping[im_name] = texture_name

        self.im_names = im_names
        self.c_names = c_names
        self.dataroot_names = dataroot_names
        self.texture_mapping = texture_mapping
        self.category_names = category_names # NEW
        # if not file with caption dict
        # self.cloth_complete_names = OrderedDict()  # {c_name: cloth_path} for each c_name in c_names
        # for idx, c_name in enumerate(c_names):
        #     cloth_path = os.path.join(
        #         dataroot, 'cleaned_inshop_imgs', c_name.replace(".jpg", "_cleaned.jpg"))
        #     if not os.path.exists(cloth_path): cloth_path = os.path.join(dataroot, 'images', c_name)
        # self.cloth_complete_names[c_name] = os.path.relpath(cloth_path, self.dataroot)

    def __getitem__(self, index: int) -> dict:
        """For each index return the corresponding element in the dataset

        Args:
            index (int): element index

        Raises:
            NotImplementedError

        Returns:
            dict:
                c_name: filename of inshop cloth
                im_name: filename of model with cloth
                cloth: img of inshop cloth
                image: img of the model with that cloth
                im_cloth: cut cloth from the model
                im_mask: black mask of the cloth in the model img
                cloth_sketch: sketch of the inshop cloth
                im_sketch: sketch of "im_cloth"
        """

        c_name = self.c_names[index]
        im_name = self.im_names[index]
        dataroot = self.dataroot_names[index]
        category = dataroot.split('/')[-1]

        sketch_threshold = random.randint(
            self.sketch_threshold_range[0], self.sketch_threshold_range[1])

        
        if "captions" in self.outputlist:
            captions = self.captions_dict[c_name.split('_')[0]]
            # if train randomly shuffle captions if there are multiple,
            # else concatenate with comma
            if self.phase == 'train':
                random.shuffle(captions) # shuffle the 3 noun chunks.
            captions = ", ".join(captions)

        if "cloth" in self.outputlist:
            # Clothing image
            cloth_path = os.path.join(
                dataroot, 'cleaned_inshop_imgs', c_name.replace(".jpg", "_cleaned.jpg"))
            if not os.path.exists(cloth_path): cloth_path = os.path.join(dataroot, 'images', c_name)
            cloth = Image.open(cloth_path)
            cloth = cloth.resize((self.width, self.height))
            cloth = self.transform(cloth)  # [-1,1]

        if "image" in self.outputlist or "im_head" in self.outputlist \
                or "im_cloth" in self.outputlist:
            image = Image.open(os.path.join(dataroot, 'images', im_name)) # dresses/images
            image = image.resize((self.width, self.height))
            image = self.transform(image)  # [-1,1]

        if "texture" in self.outputlist:
            if self.texture_order == "shuffled":
                c_texture_name = self.texture_mapping[im_name]
            else:
                c_texture_name = c_name
            textures = glob.glob(os.path.join(
                dataroot, 'textures/images', c_texture_name.replace('.jpg', ''), "*"))
            textures = sorted(textures, key=lambda x: int(str(Path(x).name).split('.')[0]))
            if self.phase == 'train':
                texture = Image.open(random.choice(textures))
            else:  # test
                texture = Image.open(textures[len(textures) // 2])

            texture = texture.resize((224, 224))  # CLIP model input size
            texture = self.transform(texture)  # [-1,1]

        if "im_sketch" in self.outputlist or "greyscale_im_sketch" in self.outputlist:
            if "unpaired" == self.order and self.phase == 'test':
                greyscale_im_sketch = Image.open(os.path.join(dataroot, 'im_sketch_unpaired',
                                                              f'{im_name.replace(".jpg", "")}_{c_name.replace(".jpg", ".png")}'))
            else:
                greyscale_im_sketch = Image.open(os.path.join(dataroot, 'im_sketch', c_name.replace(".jpg", ".png")))

            greyscale_im_sketch = greyscale_im_sketch.resize((self.width, self.height))
            greyscale_im_sketch = ImageOps.invert(greyscale_im_sketch)
            # threshold grayscale pil image
            im_sketch = greyscale_im_sketch.point(
                lambda p: 255 if p > sketch_threshold else 0)
            # im_sketch = im_sketch.convert("RGB")
            im_sketch = transforms.functional.to_tensor(im_sketch)  # [0,1]
            greyscale_im_sketch = transforms.functional.to_tensor(greyscale_im_sketch)  # [0,1]
            im_sketch = 1 - im_sketch
            greyscale_im_sketch = 1 - greyscale_im_sketch

        if "im_pose" in self.outputlist or "parser_mask" in self.outputlist \
                or "im_mask" in self.outputlist or "parse_mask_total" in \
                self.outputlist or "parse_array" in self.outputlist or \
                "pose_map" in self.outputlist or "parse_array" in \
                self.outputlist or "shape" in self.outputlist or "im_head" in self.outputlist or "inpaint_mask" in self.outputlist:
            # Label Map
            parse_name = im_name.replace('_0.jpg', '_4.png')
            im_parse = Image.open(os.path.join(
                dataroot, 'label_maps', parse_name))
            im_parse = im_parse.resize(
                (self.width, self.height), Image.NEAREST)
            parse_array = np.array(im_parse)

            parse_shape = (parse_array > 0).astype(np.float32)

            parse_head = (parse_array == 1).astype(np.float32) + \
                         (parse_array == 2).astype(np.float32) + \
                         (parse_array == 3).astype(np.float32) + \
                         (parse_array == 11).astype(np.float32)

            parser_mask_fixed = (parse_array == label_map["hair"]).astype(np.float32) + \
                                (parse_array == label_map["left_shoe"]).astype(np.float32) + \
                                (parse_array == label_map["right_shoe"]).astype(np.float32) + \
                                (parse_array == label_map["hat"]).astype(np.float32) + \
                                (parse_array == label_map["sunglasses"]).astype(np.float32) + \
                                (parse_array == label_map["scarf"]).astype(np.float32) + \
                                (parse_array == label_map["bag"]).astype(
                                    np.float32)

            parser_mask_changeable = (
                    parse_array == label_map["background"]).astype(np.float32)

            arms = (parse_array == 14).astype(np.float32) + \
                   (parse_array == 15).astype(np.float32)

            category = dataroot.split('/')[-1]
            if dataroot.split('/')[-1] == 'dresses':
                label_cat = 7
                parse_cloth = (parse_array == 7).astype(np.float32)
                parse_mask = (parse_array == 7).astype(np.float32) + \
                             (parse_array == 12).astype(np.float32) + \
                             (parse_array == 13).astype(np.float32)
                parser_mask_changeable += np.logical_and(
                    parse_array, np.logical_not(parser_mask_fixed))

            elif dataroot.split('/')[-1] == 'upper_body':
                label_cat = 4
                parse_cloth = (parse_array == 4).astype(np.float32)
                parse_mask = (parse_array == 4).astype(np.float32)

                parser_mask_fixed += (parse_array == label_map["skirt"]).astype(np.float32) + \
                                     (parse_array == label_map["pants"]).astype(
                                         np.float32)

                parser_mask_changeable += np.logical_and(
                    parse_array, np.logical_not(parser_mask_fixed))
            elif dataroot.split('/')[-1] == 'lower_body':
                label_cat = 6
                parse_cloth = (parse_array == 6).astype(np.float32)
                parse_mask = (parse_array == 6).astype(np.float32) + \
                             (parse_array == 12).astype(np.float32) + \
                             (parse_array == 13).astype(np.float32)

                parser_mask_fixed += (parse_array == label_map["upper_clothes"]).astype(np.float32) + \
                                     (parse_array == 14).astype(np.float32) + \
                                     (parse_array == 15).astype(np.float32)
                parser_mask_changeable += np.logical_and(
                    parse_array, np.logical_not(parser_mask_fixed))
            else:
                raise NotImplementedError

            parse_head = torch.from_numpy(parse_head)  # [0,1]
            parse_cloth = torch.from_numpy(parse_cloth)  # [0,1]
            parse_mask = torch.from_numpy(parse_mask)  # [0,1]
            parser_mask_fixed = torch.from_numpy(parser_mask_fixed)
            parser_mask_changeable = torch.from_numpy(parser_mask_changeable)

            parse_without_cloth = np.logical_and(
                parse_shape, np.logical_not(parse_mask))
            parse_mask = parse_mask.cpu().numpy()

            if "im_head" in self.outputlist:
                # Masked cloth
                im_head = image * parse_head - (1 - parse_head)
            if "im_cloth" in self.outputlist:
                im_cloth = image * parse_cloth + (1 - parse_cloth)

            # Shape
            parse_shape = Image.fromarray((parse_shape * 255).astype(np.uint8))
            parse_shape = parse_shape.resize(
                (self.width // 16, self.height // 16), Image.BILINEAR)
            parse_shape = parse_shape.resize(
                (self.width, self.height), Image.BILINEAR)
            shape = self.transform2d(parse_shape)  # [-1,1]

            # Load pose points
            pose_name = im_name.replace('_0.jpg', '_2.json')
            with open(os.path.join(dataroot, 'keypoints', pose_name), 'r') as f:
                pose_label = json.load(f)
                pose_data = pose_label['keypoints']
                pose_data = np.array(pose_data)
                pose_data = pose_data.reshape((-1, 4))

            point_num = pose_data.shape[0]
            pose_map = torch.zeros(point_num, self.height, self.width)
            r = self.radius * (self.height / 512.0)
            im_pose = Image.new('L', (self.width, self.height))
            pose_draw = ImageDraw.Draw(im_pose)
            neck = Image.new('L', (self.width, self.height))
            neck_draw = ImageDraw.Draw(neck)
            for i in range(point_num):
                one_map = Image.new('L', (self.width, self.height))
                draw = ImageDraw.Draw(one_map)
                point_x = np.multiply(pose_data[i, 0], self.width / 384.0)
                point_y = np.multiply(pose_data[i, 1], self.height / 512.0)
                if point_x > 1 and point_y > 1:
                    draw.rectangle((point_x - r, point_y - r,
                                    point_x + r, point_y + r), 'white', 'white')
                    pose_draw.rectangle(
                        (point_x - r, point_y - r, point_x + r, point_y + r), 'white', 'white')
                    if i == 2 or i == 5:
                        neck_draw.ellipse((point_x - r * 4, point_y - r * 4, point_x + r * 4, point_y + r * 4), 'white',
                                          'white')
                one_map = self.transform2d(one_map)
                pose_map[i] = one_map[0]

            d = []
            for pose_d in pose_data:
                ux = pose_d[0] / 384.0
                uy = pose_d[1] / 512.0

                # scale posemap points
                px = ux * self.width
                py = uy * self.height

                d.append(kpoint_to_heatmap(
                    np.array([px, py]), (self.height, self.width), 9))

            pose_map = torch.stack(d)

            # just for visualization
            im_pose = self.transform2d(im_pose)

            im_arms = Image.new('L', (self.width, self.height))
            arms_draw = ImageDraw.Draw(im_arms)
            if dataroot.split('/')[-1] == 'dresses' or dataroot.split('/')[-1] == 'upper_body' or dataroot.split('/')[
                -1] == 'lower_body':
                with open(os.path.join(dataroot, 'keypoints', pose_name), 'r') as f:
                    data = json.load(f)
                    shoulder_right = np.multiply(
                        tuple(data['keypoints'][2][:2]), self.height / 512.0)
                    shoulder_left = np.multiply(
                        tuple(data['keypoints'][5][:2]), self.height / 512.0)
                    elbow_right = np.multiply(
                        tuple(data['keypoints'][3][:2]), self.height / 512.0)
                    elbow_left = np.multiply(
                        tuple(data['keypoints'][6][:2]), self.height / 512.0)
                    wrist_right = np.multiply(
                        tuple(data['keypoints'][4][:2]), self.height / 512.0)
                    wrist_left = np.multiply(
                        tuple(data['keypoints'][7][:2]), self.height / 512.0)
                    if wrist_right[0] <= 1. and wrist_right[1] <= 1.:
                        if elbow_right[0] <= 1. and elbow_right[1] <= 1.:
                            arms_draw.line(
                                np.concatenate((wrist_left, elbow_left, shoulder_left, shoulder_right)).astype(
                                    np.uint16).tolist(), 'white', 45, 'curve')
                        else:
                            arms_draw.line(np.concatenate(
                                (wrist_left, elbow_left, shoulder_left, shoulder_right, elbow_right)).astype(
                                np.uint16).tolist(), 'white', 45, 'curve')
                    elif wrist_left[0] <= 1. and wrist_left[1] <= 1.:
                        if elbow_left[0] <= 1. and elbow_left[1] <= 1.:
                            arms_draw.line(
                                np.concatenate((shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(
                                    np.uint16).tolist(), 'white', 45, 'curve')
                        else:
                            arms_draw.line(np.concatenate(
                                (elbow_left, shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(
                                np.uint16).tolist(), 'white', 45, 'curve')
                    else:
                        arms_draw.line(np.concatenate(
                            (wrist_left, elbow_left, shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(
                            np.uint16).tolist(), 'white', 45, 'curve')

                hands = np.logical_and(np.logical_not(im_arms), arms)

                if dataroot.split('/')[-1] == 'dresses' or dataroot.split('/')[-1] == 'upper_body':
                    parse_mask += im_arms
                    parser_mask_fixed += hands

            # delete neck
            parse_head_2 = torch.clone(parse_head)
            if dataroot.split('/')[-1] == 'dresses' or dataroot.split('/')[-1] == 'upper_body':
                with open(os.path.join(dataroot, 'keypoints', pose_name), 'r') as f:
                    data = json.load(f)
                    points = []
                    points.append(np.multiply(
                        tuple(data['keypoints'][2][:2]), self.height / 512.0))
                    points.append(np.multiply(
                        tuple(data['keypoints'][5][:2]), self.height / 512.0))
                    x_coords, y_coords = zip(*points)
                    A = np.vstack([x_coords, np.ones(len(x_coords))]).T
                    m, c = np.linalg.lstsq(A, y_coords, rcond=None)[0]
                    for i in range(parse_array.shape[1]):
                        y = i * m + c
                        parse_head_2[int(
                            y - 20 * (self.height / 512.0)):, i] = 0

            parser_mask_fixed = np.logical_or(
                parser_mask_fixed, np.array(parse_head_2, dtype=np.uint16))
            parse_mask += np.logical_or(parse_mask, np.logical_and(np.array(parse_head, dtype=np.uint16),
                                                                   np.logical_not(
                                                                       np.array(parse_head_2, dtype=np.uint16))))

            # tune the amount of dilation here
            parse_mask = cv2.dilate(parse_mask, np.ones(
                (5, 5), np.uint16), iterations=5)
            parse_mask = np.logical_and(
                parser_mask_changeable, np.logical_not(parse_mask))
            parse_mask_total = np.logical_or(parse_mask, parser_mask_fixed)
            im_mask = image * parse_mask_total
            inpaint_mask = 1 - parse_mask_total

            if self.mask_type == 'bounding_box':
                # here we have to modify the mask and get the bounding box
                bboxes = masks_to_boxes(inpaint_mask.unsqueeze(0))
                bboxes = bboxes.type(torch.int32)  # xmin, ymin, xmax, ymax format
                xmin = bboxes[0, 0]
                xmax = bboxes[0, 2]
                ymin = bboxes[0, 1]
                ymax = bboxes[0, 3]

                inpaint_mask[ymin:ymax + 1, xmin:xmax + 1] = torch.logical_and(
                    torch.ones_like(inpaint_mask[ymin:ymax + 1, xmin:xmax + 1]),
                    torch.logical_not(parser_mask_fixed[ymin:ymax + 1, xmin:xmax + 1]))

                inpaint_mask = inpaint_mask.unsqueeze(0)
                im_mask = image * np.logical_not(inpaint_mask.repeat(3, 1, 1))
                parse_mask_total = parse_mask_total.numpy()
                parse_mask_total = parse_array * parse_mask_total
                parse_mask_total = torch.from_numpy(parse_mask_total)
            elif self.mask_type == "mask":
                inpaint_mask = inpaint_mask.unsqueeze(0)
                parse_mask_total = parse_mask_total.numpy()
                parse_mask_total = parse_array * parse_mask_total
                parse_mask_total = torch.from_numpy(parse_mask_total)
            else:
                raise ValueError("Unknown mask type")

        if "stitch_label" in self.outputlist:
            stitch_labelmap = Image.open(os.path.join(
                self.dataroot, 'test_stitchmap', im_name.replace(".jpg", ".png")))
            stitch_labelmap = transforms.ToTensor()(stitch_labelmap) * 255
            stitch_label = stitch_labelmap == 13

        result = {}
        for k in self.outputlist:
            result[k] = vars()[k]

        return result

    def __len__(self) -> int:
        """Return dataset length

        Returns:
            int: dataset length
        """

        return len(self.c_names)


class InShopDataset(data.Dataset):
    """Dataset for in shop garments
    """

    def __init__(self, root_dir: str, categories: Literal["dresses", "lower_body", "upper_body"] = tuple(
        ["dresses", "lower_body", "upper_body"])) -> None:
        """Initialize dataset

        Args:
            root_dir (str): path to root dir
            categories (Literal["dresses", "lower_body", "upper_body"], optional): _description_. Defaults to Tuple(["dresses", "lower_body", "upper_body"]).
        """
        self.root_dir = root_dir
        self.categories = categories
        self.cloth_names = []
        self.cat_names = []

        for cat in self.categories:
            for cloth in os.listdir(os.path.join(self.root_dir, cat, "cleaned_inshop_imgs")):
                self.cloth_names.append(os.path.join(
                    self.root_dir, cat, "cleaned_inshop_imgs", cloth))
                self.cat_names.append(cat)
        print("loaded dataset")

    def __len__(self) -> int:
        return len(self.cloth_names)

    def __getitem__(self, index: int) -> dict:
        cloth_image = Image.open(self.cloth_names[index])
        mask_filename = self.cloth_names[index].replace("inshop_imgs", "inshop_masks").replace("cleaned.jpg",
                                                                                               "cleaned_mask.png")
        mask_image = Image.open(mask_filename)
        cloth_image = transforms.ToTensor()(cloth_image)
        mask_image = transforms.ToTensor()(mask_image)
        mask_image = mask_image[0, :, :]

        return {'cloth': cloth_image, 'mask': mask_image, 'category': self.cat_names[index],
                'c_name': self.cloth_names[index][:-4]}
  
class DressCodeRetrieval(DressCodeDataset):
    def __init__(self, *args, retrieve_feat_path:str = None, n_chunks=3, 
                 chunk_combos = ["single", "double", "triplet"], 
                 retrievelist: Tuple[str] = ('retrieved_cloth', 'retrieved_cpaths'), top_k=5, 
                 shuffle=[], augment_dataset=False, chunk_ids: list = None, **kwargs):
        """_summary_

        Args:
            pred_files_paths (list, optional): precomputed retrieved files. Defaults to [].
            n_chunks (int, optional): number of chunks used per caption (max 3). Defaults to 1.
            chunks_list (list, optional): which chunk item to use. Defaults to ["single", "double", "triplet"].
            retrievelist (Tuple[str], optional): type of information to retrieve. Defaults to ('retrieved_cloth', 'retrieved_cpaths').
            top_k (int, optional): number of retrieved items per data sample. Defaults to 5.
            shuffle (list, optional): Accepts retrieved, captions. Defaults to [].
        """
        super().__init__(*args, **kwargs)

        self.pred_files = {}
        self.chunk_ids = chunk_combos
        if shuffle != []:
            assert self.phase == "train", print(f"{self.phase} phase and shuffle activated!! You can use shuffle only with train")

        # load precomputed retrieved files
        pred_files_paths = []
        if top_k > 0 and retrieve_feat_path is not None:
            for cat in self.category:
                pred_files_paths.append(os.path.join(retrieve_feat_path, f"{self.phase}_{cat}_{chunk_combos[n_chunks-1]}.json"))
        for pred_files_path in pred_files_paths:
            with open(os.path.join(pred_files_path)) as f:
                retrieved_and_metrics = json.load(f)
                self.pred_files.update(retrieved_and_metrics["pred_files"]) # self.retrieved_files
            del retrieved_and_metrics

        if top_k > 0: 
            self.pred_files = {key:value[:top_k] for key,value in self.pred_files.items()}

        # extend the dataset with noun chunks combos
        if augment_dataset:
            captions, c_names, im_names, dataroot_names = [], [], [], []
            for name, im_name, dname in zip(self.c_names, self.im_names, self.dataroot_names):
                caption = self.captions_dict[name.split("_")[0]]
                if len(set(caption)) < len(caption):
                    print(f"duplicated noun chunks at item: {name}, {caption}")

                caption = sorted(caption) # = sorted(caption) sorted(list(dict.fromkeys(caption))) example: ABB

                chunks_ids = list(combinations(range(len(caption)), min(n_chunks, len(caption))))

                all_chunks = set([", ".join([caption[idx] for idx in combo]) for combo in sorted(chunks_ids)]) # set([AB,AB,BB]) (with n_chunks=2)
                for chunk in all_chunks:
                    # if chunks not in self.pred_files: continue # necessary to ensure 3 chunks 
                    c_names.append(name), im_names.append(im_name), dataroot_names.append(dname)
                    captions.append(chunk)

        else:
            if self.chunk_ids is None: self.chunk_ids = [i for i in range(n_chunks)]
            else: assert len(self.chunk_ids) == n_chunks
            captions = []
            for c_name in self.c_names:
                chunks = ", ".join([sorted(self.captions_dict[c_name.split("_")[0]])[i] for i in range(n_chunks)])
                captions.append(chunks)
            c_names, im_names, dataroot_names = self.c_names, self.im_names, self.dataroot_names

        if self.pred_files != {}:
            assert len(set(captions)-set(self.pred_files)) == 0, print(f"these captions are not in retrieved keys: {set(captions)-set(self.pred_files)}")

        
        self.shuffle = shuffle
        self.im_names = im_names
        self.c_names = c_names
        self.dataroot_names = dataroot_names
        self.retrievelist = retrievelist
        self.n_chunks = n_chunks
        self.n_retrieved = top_k
        self.captions = captions
        
    def __len__(self):
        return len(self.c_names)


    def __getitem__(self, index: int) -> dict:
        vocab = super().__getitem__(index)
        caption = self.captions[index]

        if "captions" in self.shuffle:
            caption = ", ".join(caption.split(", ")[:random.randint(1,self.n_chunks)])
        vocab["captions"] = caption

        
        if "retrieved_cloth" in self.retrievelist:
            imgs, paths, pred_files = [], [], []
            if self.pred_files != {}:
                pred_files = self.pred_files[caption]
            else:
                pred_files = []

            if "retrieved" in self.shuffle:
                pred_files = pred_files[:random.randint(1,self.n_retrieved)]

            for file in pred_files:
                filepath = os.path.join(self.dataroot, file)
                if not os.path.exists(filepath): 
                    filepath = filepath.replace("cleaned_inshop_imgs", "images").replace("_cleaned.jpg", ".jpg")
                
                img = Image.open(filepath)
                img = img.resize((self.width, self.height))
                img = self.transform(img)  # [-1,1]
                imgs.append(img)
                paths.append(filepath)
            if len(imgs)>0: vocab["retrieved_cloth"] = torch.stack(imgs, dim=0)
            vocab["retrieved_cpaths"] = paths
        return vocab 
    
def retrieve_collate_fn(batch, var_keys, n_retrieved=0):
    # Extract the image tensors with varying first dimension (num)
    var_vocab = {}
    # iterate over the item type (eg path or image)
    for key in var_keys:
        list_key=[]
        for item in batch:
            el = item[key] # list or tensor
            # if type(el) == torch.Tensor:
            #     pad_size = n_retrieved - el.size(0) # pad only the missing retrieved images
            #     if pad_size > 0:
            #         el = F.pad(el, (0, 0, 0, 0, 0, 0, 0, pad_size), value=-0.5)
            list_key.append(el)
        if type(list_key[0]) == torch.Tensor: list_key=torch.cat(list_key, dim=0) # cat instead of stack because we don't want a new dimension
        var_vocab[key] = list_key
    
    # Extract other keys and apply the default collate_fn to them
    remaining_keys = {key: default_collate([item[key] for item in batch]) 
                      for key in batch[0] if key not in var_keys}
    
    # Return a dictionary with the images as a list and the rest collated normally
    return {**var_vocab, **remaining_keys}
