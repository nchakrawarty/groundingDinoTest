from torch.utils.data import Dataset
import os
import json
from PIL import Image

class GroundingDataset(Dataset):
    def __init__(self, annotation_file, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        with open(annotation_file, "r") as f:
            self.annotations = json.load(f)
            print("ANNOTATION DATASET", self.annotations)

    def __len__(self):
        return len(self.annotations["images"])

    def __getitem__(self, idx):
        image_info = self.annotations["images"][idx]
        image_path = os.path.join(self.image_dir, image_info["file_name"])
        image = Image.open(image_path).convert("RGB")
        
        # Get bounding boxes and labels
        annots = [ann for ann in self.annotations["annotations"] if ann["image_id"] == image_info["id"]]
        bboxes = [annot["bbox"] for annot in annots]  # [x, y, width, height]
        labels = [annot["category_id"] for annot in annots]

        if self.transform:
            image = self.transform(image)

        return image, bboxes, labels

    @staticmethod
    def collate_fn(batch):
        images, bboxes, labels = zip(*batch)
        return images, bboxes, labels

def build_dataset(annotation_file, image_dir):
    return GroundingDataset(annotation_file, image_dir)
