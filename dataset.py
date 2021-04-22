import os, torch, torchvision
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset


class COCO_detection(Dataset):
    def __init__(self, img_dir, ann, transforms=None):
        super(COCO_detection, self).__init__()
        self.img_dir = img_dir
        self.transforms = transforms
        self.coco = COCO(ann)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.label_map = {raw_label:i for i, raw_label in enumerate(self.coco.getCatIds())}

    def _load_image(self, id_):
        img = self.coco.loadImgs(id_)[0]['file_name']
        return Image.open(os.path.join(self.img_dir, img)).convert('RGB')

    def _load_target(self, id_):
        if len(self.coco.loadAnns(self.coco.getAnnIds(id_))) == 0: return None, None
        bboxs, labels = [], []
        for ann in self.coco.loadAnns(self.coco.getAnnIds(id_)):
            min_x, min_y, w, h = ann['bbox']
            bboxs.append(torch.FloatTensor([min_x, min_y, min_x+w, min_y+h]))
            labels.append(self.label_map[ann['category_id']])
        bboxs, labels = torch.stack(bboxs, 0), torch.LongTensor(labels)
        return bboxs, labels

    def __getitem__(self, index):
        id_ = self.ids[index]
        image, (bboxs, labels) = self._load_image(id_), self._load_target(id_)
        if self.transforms is not None:
            image, bboxs = self.transforms(image, bboxs)

        return image, bboxs, labels

    def __len__(self):
        return len(self.ids)
    
    
class COCO_detection_raw(Dataset):
    def __init__(self, img_dir, ann, transforms=None):
        super(COCO_detection_visualize, self).__init__()
        self.img_dir = img_dir
        self.transforms = transforms
        self.coco = COCO(ann)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def _load_image(self, id_):
        img = self.coco.loadImgs(id_)[0]['file_name']
        return Image.open(os.path.join(self.img_dir, img)).convert('RGB')

    def _load_target(self, id_):
        return self.coco.loadAnns(self.coco.getAnnIds(id_))

    def __getitem__(self, index):
        id_ = self.ids[index]
        image, target = self._load_image(id_), self._load_target(id_)
        if self.transforms is not None: image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.ids)