import os
import pickle
import random
from pathlib import Path
import glob
import json
from typing import Tuple, Literal

import cv2
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageOps
import numpy as np
# from utils.labelmap import label_map
from torchvision.ops import masks_to_boxes
from utils.posemap import get_coco_body25_mapping
from utils.posemap import kpoint_to_heatmap

PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()


# TODO: controllare la posizione che deve avere 'im_sketch_unpaired' nel dataset
class VitonHDDataset(data.Dataset):
    """VITON-HD dataset class
    """

    def __init__(self,
                 dataroot_path: str,
                 phase: Literal["train", "test"],
                 radius=5,
                 caption_file: str = 'coarse_captions_viton.json',
                 sketch_threshold_range: Tuple[int, int] = (20, 127),
                 order: Literal['paired', 'unpaired'] = 'paired',
                 outputlist: Tuple[str] = ('c_name', 'im_name', 'cloth', 'image', 'im_cloth', 'shape', 'pose_map',
                                           'parse_array', 'im_mask', 'inpaint_mask', 'parse_mask_total',
                                           'cloth_sketch', 'im_sketch', 'greyscale_im_sketch', 'captions', 'category',
                                           'skeleton', 'category', 'texture', 'parse_cloth', 'cpath'),
                 size: Tuple[int, int] = (512, 384),
                 mask_type: Literal["mask", "bounding_box"] = "bounding_box",
                 texture_order: Literal["shuffled", "original"] = "original",
                 n_chunks = 3
                 ):
        """
        Initialize the PyTroch Dataset Class
        :param dataroot_path: dataset root folder
        :type dataroot_path:  string
        :param phase: phase (train | test)
        :type phase: string
        :param order: setting (paired | unpaired)
        :type order: string
        :param category: clothing category (upper_body | lower_body | dresses)
        :type category: list(str)
        :param size: image size (height, width)
        :type size: tuple(int)
        """
        super(VitonHDDataset, self).__init__()
        self.dataroot = dataroot_path
        self.phase = phase
        self.sketch_threshold_range = sketch_threshold_range
        self.category = ('upper_body')
        self.outputlist = outputlist
        self.height = size[0]
        self.width = size[1]
        self.radius = radius
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transform2D = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.order = order
        self.mask_type = mask_type
        self.texture_order = texture_order

        im_names = []
        c_names = []
        dataroot_names = []

        possible_outputs = ['c_name', 'im_name', 'cloth', 'image', 'im_cloth', 'shape', 'im_head', 'im_pose',
                            'pose_map', 'parse_array', 'dense_labels', 'dense_uv', 'skeleton', 'im_mask',
                            'inpaint_mask', 'parse_mask_total', 'cloth_sketch', 'im_sketch', 'greyscale_im_sketch',
                            'captions', 'category', 'texture', 'parse_cloth', 'cpath']

        assert all(x in possible_outputs for x in outputlist)

        # Load Captions
        with open(os.path.join(self.dataroot, caption_file)) as f:
            self.captions_dict = json.load(f)

        dataroot = self.dataroot
        if phase == 'train':
            filename = os.path.join(dataroot, f"{phase}_pairs.txt")
        else:
            # filename = os.path.join(dataroot, f"{phase}_pairs_{order}.txt")
            # if setting is paired take two times the first part of the line
            filename = os.path.join(dataroot, f"{phase}_pairs.txt")
        texture_mapping = {}  # {cloth_name: [texture_name]} used only when texture_order == "shuffled"
        texture_filename = os.path.join(dataroot, f"test_shuffled_textures.txt")

        with open(filename, 'r') as f:
            data_len = len(f.readlines())

        with open(filename, 'r') as f:
            for line in f.readlines():
                if phase == 'train':
                    im_name, _ = line.strip().split()
                    c_name = im_name
                else:
                    if order == 'paired':
                        im_name, _ = line.strip().split()
                        c_name = im_name
                    else:
                        im_name, c_name = line.strip().split()

                im_names.append(im_name)
                c_names.append(c_name)
                dataroot_names.append(dataroot)

        if texture_order == "shuffled":
            with open(texture_filename, 'r') as f:
                for line in f.readlines():
                    im_name, texture_name = line.strip().split()
                    texture_mapping[im_name] = texture_name

        self.im_names = im_names
        self.c_names = c_names
        self.dataroot_names = dataroot_names
        self.texture_mapping = texture_mapping
        self.n_chunks = 3

    def __getitem__(self, index):
        """
        For each index return the corresponding sample in the dataset
        :param index: data index
        :type index: int
        :return: dict containing dataset samples
        :rtype: dict
        """
        c_name = self.c_names[index]
        im_name = self.im_names[index]
        dataroot = self.dataroot_names[index]
        category = 'upper_body'

        sketch_threshold = random.randint(
            self.sketch_threshold_range[0], self.sketch_threshold_range[1])

        if "captions" in self.outputlist:
            captions = self.captions_dict[c_name.split('_')[0]]
            # if train randomly shuffle captions if there are multiple,
            # else concatenate with comma
            if self.phase == 'train':
                random.shuffle(captions)
            n_chunks = min(len(captions), self.n_chunks)
            captions = captions[:n_chunks]
            captions = ", ".join(captions)

        if "cloth" in self.outputlist:
            # Clothing image
            # cloth = Image.open(os.path.join(dataroot, 'images', c_name))
            cloth = Image.open(os.path.join(
                dataroot, self.phase, 'cloth', c_name))
            cloth = cloth.resize((self.width, self.height))
            cloth = self.transform(cloth)  # [-1,1]
            cpath = os.path.join(
                self.dataroot, self.phase, 'cloth', c_name)

        if "image" in self.outputlist or "im_head" in self.outputlist or "im_cloth" in self.outputlist:
            # Person image
            # image = Image.open(os.path.join(dataroot, 'images', im_name))
            image = Image.open(os.path.join(
                dataroot, self.phase, 'image', im_name))
            image = image.resize((self.width, self.height))
            image = self.transform(image)  # [-1,1]

        if "texture" in self.outputlist:
            if self.texture_order == "shuffled":
                c_texture_name = self.texture_mapping[im_name]
            else:
                c_texture_name = c_name
            textures = glob.glob(os.path.join(
                dataroot, self.phase, 'textures', 'images', c_texture_name.replace('.jpg', ''), "*"))
            textures = sorted(textures, key=lambda x: int(str(Path(x).name).split('.')[0]))
            if self.phase == 'train':
                texture = Image.open(random.choice(textures))
            else:  # test
                texture = Image.open(textures[len(textures) // 2])

            texture = texture.resize((224, 224))  # CLIP model input size
            texture = self.transform(texture)  # [-1,1]

        if "cloth_sketch" in self.outputlist:
            # Clothing sketch image
            cloth_sketch = Image.open(
                os.path.join(dataroot, self.phase, 'cloth_sketch', c_name.replace(".jpg", ".png")))
            cloth_sketch = cloth_sketch.resize((self.width, self.height))
            cloth_sketch = ImageOps.invert(cloth_sketch)
            # threshold grayscale pil image
            cloth_sketch = cloth_sketch.point(
                lambda p: 255 if p > sketch_threshold else 0)
            cloth_sketch = cloth_sketch.convert("RGB")
            cloth_sketch = self.transform(cloth_sketch)  # [-1,1]

        if "im_sketch" in self.outputlist or "greyscale_im_sketch" in self.outputlist:
            # unpaired sketch is not present at the moment
            if "unpaired" == self.order and self.phase == 'test':
                greyscale_im_sketch = Image.open(os.path.join(dataroot, self.phase, 'im_sketch_unpaired',
                                                              f'{im_name.replace(".jpg", "")}_{c_name.replace(".jpg", ".png")}'))
            else:
                greyscale_im_sketch = Image.open(os.path.join(dataroot, self.phase, 'im_sketch', c_name.replace(".jpg", ".png")))

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

        labels = {
            0: ['background', [0, 10]],  # 0 is background, 10 is neck
            1: ['hair', [1, 2]],  # 1 and 2 are hair
            2: ['face', [4, 13]],
            3: ['upper', [5, 6, 7]],
            4: ['bottom', [9, 12]],
            5: ['left_arm', [14]],
            6: ['right_arm', [15]],
            7: ['left_leg', [16]],
            8: ['right_leg', [17]],
            9: ['left_shoe', [18]],
            10: ['right_shoe', [19]],
            11: ['socks', [8]],
            12: ['noise', [3, 11]]
        }

        if "skeleton" in self.outputlist:
            # Skeleton
            # skeleton = Image.open(os.path.join(dataroot, 'skeletons', im_name.replace("_0", "_5")))
            skeleton = Image.open(
                os.path.join(dataroot, self.phase, 'openpose_img', im_name.replace('.jpg', '_rendered.png')))
            skeleton = skeleton.resize((self.width, self.height))
            skeleton = self.transform(skeleton)

        if "im_pose" in self.outputlist or "parser_mask" in self.outputlist or "im_mask" in self.outputlist \
                or "parse_mask_total" in self.outputlist or "parse_array" in self.outputlist \
                or "pose_map" in self.outputlist or "parse_array" in self.outputlist or "shape" in self.outputlist \
                or "im_head" in self.outputlist or "inpaint_mask" in self.outputlist:
            # Label Map
            # parse_name = im_name.replace('_0.jpg', '_4.png')
            parse_name = im_name.replace('.jpg', '.png')
            im_parse = Image.open(os.path.join(
                dataroot, self.phase, 'image-parse-v3', parse_name))
            im_parse = im_parse.resize(
                (self.width, self.height), Image.NEAREST)
            im_parse_final = transforms.ToTensor()(im_parse) * 255
            parse_array = np.array(im_parse)

            parse_shape = (parse_array > 0).astype(np.float32)

            parse_head = (parse_array == 1).astype(np.float32) + \
                         (parse_array == 2).astype(np.float32) + \
                         (parse_array == 4).astype(np.float32) + \
                         (parse_array == 13).astype(np.float32)

            parser_mask_fixed = (parse_array == 1).astype(np.float32) + \
                                (parse_array == 2).astype(np.float32) + \
                                (parse_array == 18).astype(np.float32) + \
                                (parse_array == 19).astype(np.float32)

            parser_mask_changeable = (parse_array == 0).astype(np.float32)

            arms = (parse_array == 14).astype(np.float32) + \
                   (parse_array == 15).astype(np.float32)

            parse_cloth = (parse_array == 5).astype(np.float32) + \
                          (parse_array == 6).astype(np.float32) + \
                          (parse_array == 7).astype(np.float32)
            parse_mask = (parse_array == 5).astype(np.float32) + \
                         (parse_array == 6).astype(np.float32) + \
                         (parse_array == 7).astype(np.float32)

            parser_mask_fixed = parser_mask_fixed + (parse_array == 9).astype(np.float32) + \
                                (parse_array == 12).astype(np.float32)  # the lower body is fixed

            parser_mask_changeable += np.logical_and(
                parse_array, np.logical_not(parser_mask_fixed))

            parse_head = torch.from_numpy(parse_head)  # [0,1]
            parse_cloth = torch.from_numpy(parse_cloth)  # [0,1]
            parse_mask = torch.from_numpy(parse_mask)  # [0,1]
            parser_mask_fixed = torch.from_numpy(parser_mask_fixed)
            parser_mask_changeable = torch.from_numpy(parser_mask_changeable)

            # dilation
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
            shape = self.transform2D(parse_shape)  # [-1,1]

            # Load pose points
            pose_name = im_name.replace('.jpg', '_keypoints.json')
            with open(os.path.join(dataroot, self.phase, 'openpose_json', pose_name), 'r') as f:
                pose_label = json.load(f)
                pose_data = pose_label['people'][0]['pose_keypoints_2d']
                pose_data = np.array(pose_data)
                pose_data = pose_data.reshape((-1, 3))[:, :2]

                # rescale keypoints on the base of height and width
                pose_data[:, 0] = pose_data[:, 0] * (self.width / 768)
                pose_data[:, 1] = pose_data[:, 1] * (self.height / 1024)

            pose_mapping = get_coco_body25_mapping()
            # NB capire come fare per il neck visto che avrei gia la label volendo

            # point_num = pose_data.shape[0]
            point_num = len(pose_mapping)

            pose_map = torch.zeros(point_num, self.height, self.width)
            r = self.radius * (self.height / 512.0)
            im_pose = Image.new('L', (self.width, self.height))
            pose_draw = ImageDraw.Draw(im_pose)
            neck = Image.new('L', (self.width, self.height))
            neck_draw = ImageDraw.Draw(neck)
            for i in range(point_num):
                one_map = Image.new('L', (self.width, self.height))
                draw = ImageDraw.Draw(one_map)

                point_x = np.multiply(pose_data[pose_mapping[i], 0], 1)
                point_y = np.multiply(pose_data[pose_mapping[i], 1], 1)

                if point_x > 1 and point_y > 1:
                    draw.rectangle((point_x - r, point_y - r,
                                    point_x + r, point_y + r), 'white', 'white')
                    pose_draw.rectangle(
                        (point_x - r, point_y - r, point_x + r, point_y + r), 'white', 'white')
                    if i == 2 or i == 5:
                        neck_draw.ellipse((point_x - r * 4, point_y - r * 4, point_x + r * 4, point_y + r * 4), 'white',
                                          'white')
                one_map = self.transform2D(one_map)
                pose_map[i] = one_map[0]

            d = []

            for idx in range(point_num):
                ux = pose_data[pose_mapping[idx], 0]  # / (192)
                uy = (pose_data[pose_mapping[idx], 1])  # / (256)

                # scale posemap points
                px = ux  # * self.width
                py = uy  # * self.height

                d.append(kpoint_to_heatmap(
                    np.array([px, py]), (self.height, self.width), 9))

            pose_map = torch.stack(d)

            # just for visualization
            im_pose = self.transform2D(im_pose)

            im_arms = Image.new('L', (self.width, self.height))
            arms_draw = ImageDraw.Draw(im_arms)

            # do in any case because i have only upperbody
            with open(os.path.join(dataroot, self.phase, 'openpose_json', pose_name), 'r') as f:
                data = json.load(f)
                data = data['people'][0]['pose_keypoints_2d']
                data = np.array(data)
                data = data.reshape((-1, 3))[:, :2]

                # rescale keypoints on the base of height and width
                data[:, 0] = data[:, 0] * (self.width / 768)
                data[:, 1] = data[:, 1] * (self.height / 1024)

                shoulder_right = tuple(data[pose_mapping[2]])
                shoulder_left = tuple(data[pose_mapping[5]])
                elbow_right = tuple(data[pose_mapping[3]])
                elbow_left = tuple(data[pose_mapping[6]])
                wrist_right = tuple(data[pose_mapping[4]])
                wrist_left = tuple(data[pose_mapping[7]])

                ARM_LINE_WIDTH = int(90 / 512 * self.height)
                if wrist_right[0] <= 1. and wrist_right[1] <= 1.:
                    if elbow_right[0] <= 1. and elbow_right[1] <= 1.:
                        arms_draw.line(
                            np.concatenate((wrist_left, elbow_left, shoulder_left, shoulder_right)).astype(
                                np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
                    else:
                        arms_draw.line(np.concatenate(
                            (wrist_left, elbow_left, shoulder_left, shoulder_right, elbow_right)).astype(
                            np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
                elif wrist_left[0] <= 1. and wrist_left[1] <= 1.:
                    if elbow_left[0] <= 1. and elbow_left[1] <= 1.:
                        arms_draw.line(
                            np.concatenate((shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(
                                np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
                    else:
                        arms_draw.line(np.concatenate(
                            (elbow_left, shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(
                            np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
                else:
                    arms_draw.line(np.concatenate(
                        (wrist_left, elbow_left, shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(
                        np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')

                hands = np.logical_and(np.logical_not(im_arms), arms)
                parse_mask += im_arms
                parser_mask_fixed += hands

            # delete neck
            parse_head_2 = torch.clone(parse_head)

            parser_mask_fixed = np.logical_or(
                parser_mask_fixed, np.array(parse_head_2, dtype=np.uint16))
            parse_mask += np.logical_or(parse_mask, np.logical_and(np.array(parse_head, dtype=np.uint16),
                                                                   np.logical_not(
                                                                       np.array(parse_head_2, dtype=np.uint16))))

            if self.mask_type == "bounding_box":
                parse_mask = np.logical_and(
                    parser_mask_changeable, np.logical_not(parse_mask))
                parse_mask_total = np.logical_or(parse_mask, parser_mask_fixed)
                # im_mask = image * parse_mask_total
                inpaint_mask = 1 - parse_mask_total

                # here we have to modify the mask and get the bounding box
                bboxes = masks_to_boxes(inpaint_mask.unsqueeze(0))
                # xmin, ymin, xmax, ymax format
                bboxes = bboxes.type(torch.int32)
                xmin = bboxes[0, 0]
                xmax = bboxes[0, 2]
                ymin = bboxes[0, 1]
                ymax = bboxes[0, 3]

                inpaint_mask[ymin:ymax + 1, xmin:xmax + 1] = torch.logical_and(
                    torch.ones_like(
                        inpaint_mask[ymin:ymax + 1, xmin:xmax + 1]),
                    torch.logical_not(parser_mask_fixed[ymin:ymax + 1, xmin:xmax + 1]))

                inpaint_mask = inpaint_mask.unsqueeze(0)
                im_mask = image * np.logical_not(inpaint_mask.repeat(3, 1, 1))
                parse_mask_total = parse_mask_total.numpy()
                parse_mask_total = parse_array * parse_mask_total
                parse_mask_total = torch.from_numpy(parse_mask_total)
            elif self.mask_type == "keypoints":
                # tune the amount of dilation here
                parse_mask = cv2.dilate(parse_mask, np.ones(
                    (5, 5), np.uint16), iterations=5)
                parse_mask = np.logical_and(
                    parser_mask_changeable, np.logical_not(parse_mask))
                parse_mask_total = np.logical_or(parse_mask, parser_mask_fixed)
                im_mask = image * parse_mask_total
                inpaint_mask = 1 - parse_mask_total
                inpaint_mask = inpaint_mask.unsqueeze(0)
                parse_mask_total = parse_mask_total.numpy()
                parse_mask_total = parse_array * parse_mask_total
                parse_mask_total = torch.from_numpy(parse_mask_total)
            else:
                raise NotImplementedError

        if "dense_uv" in self.outputlist:
            uv = np.load(os.path.join(dataroot, 'dense',
                                      im_name.replace('_0.jpg', '_5_uv.npz')))
            uv = uv['uv']
            uv = torch.from_numpy(uv)
            uv = transforms.functional.resize(uv, (self.height, self.width))

        if "dense_labels" in self.outputlist:
            labels = Image.open(os.path.join(
                dataroot, 'dense', im_name.replace('_0.jpg', '_5.png')))
            labels = labels.resize((self.width, self.height), Image.NEAREST)
            labels = np.array(labels)

        result = {}
        for k in self.outputlist:
            result[k] = vars()[k]

        # result['im_parse'] = im_parse_final
        # result['hands'] = torch.from_numpy(hands)

        # Output interpretation
        # "c_name" -> filename of inshop cloth
        # "im_name" -> filename of model with cloth
        # "cloth" -> img of inshop cloth
        # "image" -> img of the model with that cloth
        # "im_cloth" -> cut cloth from the model
        # "im_mask" -> black mask of the cloth in the model img
        # "cloth_sketch" -> sketch of the inshop cloth
        # "im_sketch" -> sketch of "im_cloth"

        return result

    def __len__(self):
        return len(self.c_names)


class InShopDataset(data.Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.cloth_names = []
        self.parent_names = []
        phase = ["train", "test"]

        for p in phase:
            for cloth in os.listdir(os.path.join(self.root_dir, p, "cloth")):
                self.cloth_names.append(os.path.join(
                    self.root_dir, p, "cloth", cloth))
                self.parent_names.append(p)
        print("loaded dataset")

    def __len__(self):
        return len(self.cloth_names)

    def __getitem__(self, index):
        cloth_image = Image.open(self.cloth_names[index])
        mask_filename = self.cloth_names[index].replace("cloth", "cloth-mask-ours")
        mask_image = Image.open(mask_filename)
        cloth_image = transforms.ToTensor()(cloth_image)
        mask_image = transforms.ToTensor()(mask_image)
        mask_image = mask_image[0, :, :]

        return {'cloth': cloth_image, 'mask': mask_image, 'parent': self.parent_names[index],
                'c_name': self.cloth_names[index][:-4]}
