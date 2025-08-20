from torch.utils.data import Dataset
from PIL import Image
import json, os

class VQADataset(Dataset):
    def __init__(self, root, ann_file, transform=None):
        root = "data/images"
        ann_file = "data/questions.json"
        with open(ann_file, "r") as f:
            self.annotations = json.load(f)
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_path = os.path.join(self.root, ann["image"])
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, ann["question"], ann["answers"]