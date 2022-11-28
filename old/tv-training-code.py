# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import os
import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.faster_rcnn import GeneralizedRCNNTransform

from engine import train_one_epoch, evaluate
import utils
# import transforms as T
import h5py
from torchvision import transforms as T


class PennFudanDataset(object):
    def __init__(self, transforms, train=False, val=False):
        assert ((train and not val) or (val and not train))

        self._archives = None
        self.dir_to_dataset = '/home/neptun/PycharmProjects/dataset/dota'
        if train==True:
            id_all = np.load(os.path.join(self.dir_to_dataset, 'train_id.npy')).tolist()
        if val==True:
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
        image = self.archives.get('img_'+path)[:]
        boxes = self.archives.get('box_'+path)[:]
        label = self.archives.get('label_'+path)[:]
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
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 18+1
    # use our dataset and defined transformations
    dataset = PennFudanDataset(transforms=get_transform(), train=True)
    dataset_test = PennFudanDataset(transforms=get_transform(), val=True)

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    # dataset = torch.utils.data.Subset(dataset, indices[:-50])
    # dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=32, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(params, lr=0.005,
    #                             momentum=0.9, weight_decay=0.0005)
    optimizer = torch.optim.Adam(params, lr=1e-5)
    # and a learning rate scheduler
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
    #                                                step_size=3,
    #                                                gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=100)
        # update the learning rate
        # lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)
        torch.save(model.state_dict(), 'old/models/model0.pth')

    print("That's it!")
    
if __name__ == "__main__":
    # dataset = PennFudanDataset(transforms=get_transform(), val=True)
    # data_loader = torch.utils.data.DataLoader(
    #     dataset, batch_size=2, shuffle=True, num_workers=4,
    #     collate_fn=utils.collate_fn)
    #
    # iterdl = iter(data_loader)
    # batch = next(iterdl)
    # for i in range(2):
    #     fig, ax = plt.subplots()
    #     foto = batch[0][i]
    #     foto = np.moveaxis(foto.numpy(), 0, 2)
    #     ax.imshow(foto)
    #     rec = batch[1][i]['boxes']
    #     for j in range(rec.shape[0]):
    #         rect = patches.Rectangle((rec[j, 1], rec[j, 0]),
    #                                  (rec[j, 3] - rec[j, 1]),
    #                                  (rec[j, 2] - rec[j, 0]), linewidth=1, edgecolor='r', facecolor='none')
    #         ax.add_patch(rect)
    #     plt.show()


    main()
