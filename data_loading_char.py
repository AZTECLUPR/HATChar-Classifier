import os
import json
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from torchvision.transforms import RandomAffine
import torch
import random

def strokes_to_image(points, image_size=128, stroke_width=2):
    if not points:
        return Image.new('L', (image_size, image_size), color=255)
    xs = [pt['x'] for pt in points]
    ys = [pt['y'] for pt in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    range_x = max(max_x - min_x, 1e-5)
    range_y = max(max_y - min_y, 1e-5)
    norm_points = [
        {
            'x': int((pt['x'] - min_x) / range_x * (image_size - 1)),
            'y': int((pt['y'] - min_y) / range_y * (image_size - 1)),
            'stroke': pt['stroke']
        }
        for pt in points
    ]
    img = Image.new('L', (image_size, image_size), color=255)
    draw = ImageDraw.Draw(img)
    strokes = {}
    for pt in norm_points:
        strokes.setdefault(pt['stroke'], []).append((pt['x'], pt['y']))
    for stroke_points in strokes.values():
        if len(stroke_points) > 1:
            draw.line(stroke_points, fill=0, width=stroke_width)
        else:
            x, y = stroke_points[0]
            draw.ellipse([x-1, y-1, x+1, y+1], fill=0)
    return img

class StrokeSegmentDataset(Dataset):
    def __init__(self, split_folder, augmentation_probability=0.5, seed=42, image_size=224, input_mode="both"):
        self.split_folder = split_folder
        self.augmentation_probability = augmentation_probability
        self.seed = seed
        self.image_size = image_size
        self.input_mode = input_mode

        random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.data = self._load_json_files()

    def _load_json_files(self):
        # ✅ Flat folder — no subfolders
        return sorted([
            os.path.join(self.split_folder, f)
            for f in os.listdir(self.split_folder)
            if f.endswith('.json')
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        json_path = self.data[idx]
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        points = data['points']
        label = data['substring']

        # STROKE PROCESSING
        if self.input_mode in ["both", "stroke"]:
            strokes_dict = {}
            for pt in points:
                strokes_dict.setdefault(pt['stroke'], []).append([pt['x'], pt['y']])
            stroke_list = [strokes_dict[k] for k in sorted(strokes_dict.keys())]

            strokes_with_penstate = []
            for stroke in stroke_list:
                for i, xy in enumerate(stroke):
                    pen_state = 1 if i == len(stroke) - 1 else 0
                    strokes_with_penstate.append(xy + [pen_state])
            strokes = torch.tensor(strokes_with_penstate, dtype=torch.float32)
            
            # Augmentation
            random.seed(self.seed + idx)
            torch.manual_seed(self.seed + idx)
            if self.input_mode != "image" and random.random() < self.augmentation_probability:
                transform_params = random.uniform(0.9, 1.1)
                strokes = self._apply_synchronized_augmentation(strokes, transform_params)
        else:
            strokes = None

        # IMAGE PROCESSING
        if self.input_mode in ["both", "image"]:
            image = strokes_to_image(points, image_size=self.image_size)
            image = F.to_tensor(image)

            random.seed(self.seed + idx)
            torch.manual_seed(self.seed + idx)
            if self.input_mode != "stroke" and random.random() < self.augmentation_probability:
                transform_params = random.uniform(0.9, 1.1)
                affine_transform = RandomAffine(
                    degrees=15, translate=(0.1, 0.1),
                    scale=(transform_params, transform_params)
                )
                image = affine_transform(image)
        else:
            image = None

        return strokes, image, label

    def _apply_synchronized_augmentation(self, strokes, transform_factor):
        strokes[:, 0] *= transform_factor
        strokes[:, 1] *= transform_factor
        jitter = torch.randn_like(strokes) * 0.01
        strokes += jitter
        return strokes

def custom_collate_fn(batch):
    strokes, images, labels = zip(*batch)

    # Collate strokes if available
    if strokes[0] is not None:
        max_stroke_len = max(s.size(0) for s in strokes)
        padded_strokes = torch.zeros((len(strokes), max_stroke_len, 3))
        stroke_mask = torch.zeros(len(strokes), max_stroke_len, dtype=torch.bool)
        for i, s in enumerate(strokes):
            padded_strokes[i, :s.size(0), :] = s
            stroke_mask[i, :s.size(0)] = True
    else:
        padded_strokes = None
        stroke_mask = None

    images = torch.stack(images, dim=0) if images[0] is not None else None

    return padded_strokes, images, labels, stroke_mask