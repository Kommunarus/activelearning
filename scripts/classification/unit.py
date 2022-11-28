import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms

import torch
import uuid
import h5py
from PIL import Image

import random
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


mean = [0.4914, 0.4822, 0.4465]
std = [0.2470, 0.2435, 0.2616]

data_transform = transforms.Compose([
    # transforms.ToPILImage(),
    # transforms.Resize(224),
    transforms.ToTensor(),
    # transforms.Normalize(mean=mean,  std=std)
])


class NeuralNetwork(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.fc1 = nn.Linear(1280, 640)
        self.fc2 = nn.Linear(640, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.relu = nn.ReLU()

        self.sm = nn.Softmax(1)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        y = self.relu(self.fc1(x))
        hid = self.relu(self.fc2(y))
        out = self.fc3(hid)
        prob = self.sm(out)

        return hid, out, prob


class NeuralNetwork_transfer(nn.Module):  # inherit pytorch's nn.Module
    """Simple model to predict whether an item will be classified correctly
    """

    def __init__(self, input_layer):
        super(NeuralNetwork_transfer, self).__init__()  # call parent init

        self.fc1 = nn.Linear(input_layer, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 2)

        self.relu = nn.ReLU()

        self.sm = nn.Softmax(1)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        # Define how data is passed through the model and what gets returned

        y = self.relu(self.fc1(x))
        y2 = self.relu(self.fc2(y))
        out = self.fc3(y2)
        prob = self.sm(out)

        return out, prob



class Dataset_plant(Dataset):
    def __init__(self, limit=-1, transform=None, type='', step=0, al='', exclude=[], del_labels=[]):
        '''
        :param al:  "train_model", "find_score"
        :param type:  "train", "val", "test"
        '''
        self.dir_to_dataset = '/home/neptun/PycharmProjects/dataset/dataset_plant'
        self.transform = transform

        if type == 'val':
            val = True
            train = False
            test = False
        elif type == 'train':
            val = False
            train = True
            test = False
        else:
            val = False
            train = False
            test = True

        self._archives = h5py.File(os.path.join(self.dir_to_dataset, 'data.hdf5'), 'r')

        label_superclass1 = np.load(os.path.join(self.dir_to_dataset, 'label.npy')).tolist()
        filenames1 = np.load(os.path.join(self.dir_to_dataset, 'id.npy')).tolist()

        le = LabelEncoder()
        le.fit(label_superclass1)
        label_superclass1 = le.transform(label_superclass1).tolist()
        self.classes = le.classes_

        if len(exclude) > 0:
            list_indx = []
            for itt in exclude:
                indx = filenames1.index(itt)
                list_indx.append(indx)
            list_indx.sort(reverse=True)
            for indx in list_indx:
                label_superclass1.pop(indx)
                filenames1.pop(indx)

        label_superclass_train, label_superclass_val, filenames_train, \
            filenames_val = train_test_split(label_superclass1, filenames1,
                                              test_size = 0.1, random_state = 42)
        images1 = None
        label_superclass1 = None
        filenames1 = None
        if train:
            label_superclass = label_superclass_train
            filenames = filenames_train

            if len(del_labels) > 0:
                list_indx = []
                for i, row in enumerate(label_superclass):
                    if row in del_labels:
                        list_indx.append(i)
                list_indx.sort(reverse=True)
                for indx in list_indx:
                    label_superclass.pop(indx)
                    filenames.pop(indx)


        # elif test:
        #     images = images_test
        #     label_superclass = label_superclass_test
        #     filenames = filenames_test
        else:
            label_superclass = label_superclass_val
            filenames = filenames_val

        N = len(filenames)
        indx_train = list(range(N))

        add_indx = []
        if step > 0 and limit < 0:
            add_name = []
            for st in range(1, step + 1):
                with open(f'./models/step{st}.txt') as f:
                    dopindx = f.readline()
                    arr = dopindx.split(';')
                    arr_str = [x.strip() for x in arr]
                    add_name = arr_str + add_name
            add_indx = [filenames.index(x) for x in add_name]


        if val:
            self.labels = label_superclass
            self.filenames = filenames
        else:
            if limit > 0:
                indx = random.sample(indx_train, k=limit)
            elif al == 'train_model':
                indx = add_indx
            elif al == 'find_score':
                indx = list(set(range(N)) - set(add_indx))
            else:
                indx = indx_train
            self.labels = [label_superclass[x] for x in indx]
            self.filenames = [filenames[x] for x in indx]

        self._archives = None


    def __len__(self):
        return len(self.filenames)

    @property
    def archives(self):
        if self._archives is None:  # lazy loading here!
            self._archives = h5py.File(os.path.join(self.dir_to_dataset, 'data.hdf5'), 'r')
        return self._archives

    def __getitem__(self, idx):
        lab = self.labels[idx]
        img_path = self.filenames[idx]
        image = self.archives.get(img_path)[:]
        if self.transform:
            image = self.transform(image)
        return (image, lab, img_path)


class Dataset_plants_free(Dataset):
    def __init__(self, ids, dir_to_dataset, model_feacher, device, path_numpy, path_to_dir_train='', vae=False):
        self.dir_to_dataset = dir_to_dataset
        self.model_feacher = model_feacher
        self.device = device
        self.path_numpy = path_numpy
        self.vae = vae

        label_all = os.listdir(self.dir_to_dataset)

        le = LabelEncoder()
        le.fit(label_all)
        code_label = le.transform(label_all).tolist()
        id_label = {}
        for label in label_all:
            files = os.listdir(os.path.join(self.dir_to_dataset, label))
            for file in files:
                id_label[file] = (label, code_label[label_all.index(label)])

        if len(ids) > len(id_label):
            label_all = os.listdir(path_to_dir_train)

            for label in label_all:
                files = os.listdir(os.path.join(path_to_dir_train, label))
                for file in files:
                    id_label[file] = (label, code_label[label_all.index(label)])

        self.id_label = id_label
        self.ids = ids
        self.transform = data_transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]
        # path_to_f = os.path.join('/home/neptun/PycharmProjects/datasets/numpy', id)
        path_to_f = os.path.join(self.path_numpy, id)
        path_to_f2 = path_to_f + '.npy'
        label = self.id_label[id][1]
        label_str = self.id_label[id][0]
        if os.path.exists(path_to_f2):
            fea = np.load(path_to_f2)
            fea = torch.from_numpy(fea).to(self.device).to(torch.float32)
        else:
            # image = self.archives.get(id)[:]
            image = Image.open(os.path.join(self.dir_to_dataset, label_str, id))
            if self.vae:
                image = image.resize((64, 64))
            else:
                image = image.resize((224, 224))

            if len(image.size) == 2:
                image = image.convert('RGB')

            np_img = np.array(image)

            if self.transform:
                image = self.transform(np_img)

            image = image.to(self.device)
            image = torch.unsqueeze(image, 0)

            fea = self.model_feacher.predict(image)
            if self.vae:
                fea = torch.cat([fea[0], fea[1]], 1)
            fea = fea.squeeze()

            np.save(path_to_f, fea.tolist())


        return (fea, label, id)


def prepare_items(limit=-1, del_labels=[], train=False, val=False, files=None, test_size=0.1):

    assert ((train and not val) or (val and not train))

    label_all = []
    id_all = []
    for k, v in files.items():
        id_all.append(k)
        label_all.append(v[1])

    if test_size > 0:
        label_superclass_train, label_superclass_val, filenames_train, \
        filenames_val = train_test_split(label_all, id_all,
                                         test_size=test_size, random_state=42)
    else:
        label_superclass_train = label_all
        label_superclass_val = []
        filenames_train = id_all
        filenames_val = []



    if train:
        label_superclass = label_superclass_train
        filenames = filenames_train

        if len(del_labels) > 0:
            list_indx = []
            for i, row in enumerate(label_superclass):
                if row in del_labels:
                    list_indx.append(i)
            list_indx.sort(reverse=True)
            for indx in list_indx:
                filenames.pop(indx)
    if val:
        filenames = filenames_val
        if len(del_labels) > 0:
            list_indx = []
            for i, row in enumerate(label_superclass_val):
                if row in del_labels:
                    list_indx.append(i)
            list_indx.sort(reverse=True)
            for indx in list_indx:
                filenames.pop(indx)

    indx_train = list(range(len(filenames)))
    if isinstance(limit, int):
        if limit > 0:

            # random.seed(42)
            indx = random.sample(indx_train, k=limit)

            filenames = [filenames[x] for x in indx]
        elif limit == 0:
            filenames = []
    else:
        limit_arr = [int(x) for x in limit.split(':')]
        indx = random.sample(indx_train, k=limit_arr[1])[limit_arr[0]:]

        filenames = [filenames[x] for x in indx]

    return filenames


def read_dirs_dataset(dir_to_dataset):
    # dir_to_dataset = '/home/neptun/PycharmProjects/dataset/dataset_plant'

    # label_all = np.load(os.path.join(dir_to_dataset, 'label.npy')).tolist()
    # id_all = np.load(os.path.join(dir_to_dataset, 'id.npy')).tolist()
    label_all = os.listdir(dir_to_dataset)

    le = LabelEncoder()
    le.fit(label_all)
    code_label = le.transform(label_all).tolist()
    id_label = {}
    for label in label_all:
        files = os.listdir(os.path.join(dir_to_dataset, label))
        for file in files:
            id_label[file] = (label, code_label[label_all.index(label)])

    # id_label = {x: (y, z) for x, y, z in zip(id_all, label_all, code_label)}

    return id_label, len(label_all)


class Dataset_transfer_learning(Dataset):
    def __init__(self, item_hidden_layers, correct_predictions, incorrect_predictions):
        self.id = [x for x in item_hidden_layers.keys()]
        self.data = [x for x in item_hidden_layers.values()]
        self.correct_predictions = correct_predictions
        self.incorrect_predictions = incorrect_predictions

    def __len__(self):
        return len(self.id)

    def __getitem__(self, item):
        data = self.data[item]
        if self.id[item] in self.correct_predictions:
            label = 1
        else:
            label = 0
        return data, label


# Object detection
def prepare_items_od(limit=-1, train=False, val=False, seed=None, lable=None):

    assert ((train and not val) or (val and not train))

    dir_to_dataset = '/home/neptun/PycharmProjects/datasets/dota'
    if train:
        id_all = np.load(os.path.join(dir_to_dataset, 'train_id.npy')).tolist()
    if val:
        id_all = np.load(os.path.join(dir_to_dataset, 'val_id.npy')).tolist()
    if train:
        archives = h5py.File(os.path.join(dir_to_dataset, 'train_data.hdf5'), 'r')
    if val:
        archives = h5py.File(os.path.join(dir_to_dataset, 'val_data.hdf5'), 'r')

    if not lable is None:
        list_indx = []
        for i, idx in enumerate(id_all):
            if lable in archives.get('label_' + idx)[:]:
                list_indx.append(idx)
        id_all_labels = list_indx
        id_all_notlabels = list(set(id_all) - set(list_indx))

    # print(len(id_all))
    if limit > 0:
        random.seed(seed)
        if lable is None:
            indx_train = list(range(len(id_all)))
            indx = random.sample(indx_train, k=int(limit * len(id_all)))
            id_all = [id_all[x] for x in indx]
        else:
            indx_lab = list(range(len(id_all_labels)))
            indx_lab = random.sample(indx_lab, k=int(limit * len(indx_lab)))
            indx_notlab = list(range(len(id_all_notlabels)))
            indx_notlab = random.sample(indx_notlab, k=len(indx_lab))
            id_all = [id_all[x] for x in indx_lab] + [id_all[x] for x in indx_notlab]
    else:
        random.seed(seed)
        if lable is None:
            indx_train = list(range(len(id_all)))
            id_all = [id_all[x] for x in indx_train]
        else:
            indx_lab = list(range(len(id_all_labels)))
            indx_notlab = list(range(len(id_all_notlabels)))
            indx_notlab = random.sample(indx_notlab, k=min(len(indx_lab), len(id_all_notlabels)))
            id_all = [id_all[x] for x in indx_lab] + [id_all[x] for x in indx_notlab]



    return id_all




if __name__ == '__main__':
    ds = Dataset_plant(type='train')
    iter1 = iter(ds)
    photo = next(iter1)
    plt.imshow(np.moveaxis(photo[0], 0, 2))
    plt.show()
    photo = next(iter1)
    plt.imshow(np.moveaxis(photo[0], 0, 2))
    plt.show()
    photo = next(iter1)
    plt.imshow(np.moveaxis(photo[0], 0, 2))
    plt.show()
