import os
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, image_dir, annot_dir, transforms=None):
        self.image_dir = image_dir
        self.annot_dir = annot_dir
        self.transforms = transforms
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def parse_voc_xml(self, xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        boxes = []
        labels = []
        
        for obj in root.findall("object"):
            label = obj.find("name").text
            if label in self.LABELS:
                labels.append(self.LABELS[label])  # Convert label name to label ID

            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])

        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        annot_path = os.path.join(self.annot_dir, img_name.replace(".jpg", ".xml"))

        # Load image
        img = Image.open(img_path).convert("RGB")

        # Load annotations
        boxes, labels = self.parse_voc_xml(annot_path)
        
        if boxes.shape[0] == 0:
            return None

        target = {
            "boxes": boxes,
            "labels": labels
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target