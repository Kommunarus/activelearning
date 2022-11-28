# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.faster_rcnn import GeneralizedRCNNTransform
from torchvision.ops.boxes import nms

import utils
# import transforms as T
import h5py
from torchvision import transforms as T


class PennFudanDataset(object):
    def __init__(self, transforms, train=False, val=False):
        assert ((train and not val) or (val and not train))

        self._archives = None
        self.dir_to_dataset = '/home/neptun/PycharmProjects/dataset/dota'
        if train == True:
            id_all = np.load(os.path.join(self.dir_to_dataset, 'train_id.npy')).tolist()
        if val == True:
            id_all = np.load(os.path.join(self.dir_to_dataset, 'val_id.npy')).tolist()

        self.ids = id_all
        self.transforms = transforms
        self.train = train
        self.val = val

    @property
    def archives(self):
        if self._archives is None:  # lazy loading here!
            if self.train:
                self._archives = h5py.File(os.path.join(self.dir_to_dataset, 'train_data.hdf5'), 'r')
            if self.val:
                self._archives = h5py.File(os.path.join(self.dir_to_dataset, 'val_data.hdf5'), 'r')
        return self._archives

    def __getitem__(self, idx):
        path = self.ids[idx]
        image = self.archives.get('img_' + path)[:]
        boxes = self.archives.get('box_' + path)[:]
        label = self.archives.get('label_' + path)[:]
        num_objs = label.size
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(label + 1, dtype=torch.int64)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.as_tensor(idx, dtype=torch.int64)
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            image = self.transforms(image)

        return image, target

    def __len__(self):
        return len(self.ids)


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.transform = GeneralizedRCNNTransform(min_size=224, max_size=224, image_mean=[0.485, 0.456, 0.406],
                                               image_std=[0.229, 0.224, 0.225])

    return model


def get_transform():
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)


def main():
    iou_threshold = 0.5
    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
    cpu_device = torch.device("cpu")
    num_classes = 18 + 1
    dataset_test = PennFudanDataset(transforms=get_transform(), val=True)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=8, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)
    model.load_state_dict(torch.load('./models/model0.pth'))
    model.to(device)
    model.eval()

    # evaluate(model, data_loader_test, device=device)

    iterdl = iter(data_loader_test)
    images, targets = next(iterdl)
    images2 = list(img.to(device) for img in images)
    with torch.no_grad():
        outputs = model(images2)
    outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
    for i in range(8):
        fig, ax = plt.subplots()
        foto = np.moveaxis(images[i].numpy(), 0, 2)
        ax.imshow(foto)
        boxes = outputs[i]['boxes'].to(cpu_device)
        scores = outputs[i]['scores'].to(cpu_device)
        ind = nms(boxes, scores, iou_threshold).detach().cpu().numpy()
        for j in ind:
            if scores[j] > 0.5:
                rect = patches.Rectangle((boxes[j, 0], boxes[j, 1]),
                                             (boxes[j, 2] - boxes[j, 0]),
                                             (boxes[j, 3] - boxes[j, 1]), linewidth=2, edgecolor='g', facecolor='none')
            else:
                rect = patches.Rectangle((boxes[j, 0], boxes[j, 1]),
                                             (boxes[j, 2] - boxes[j, 0]),
                                             (boxes[j, 3] - boxes[j, 1]), linewidth=0.5, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
    plt.show()


if __name__ == "__main__":
    main()
