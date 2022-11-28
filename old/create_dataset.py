import os
from PIL import Image
import numpy as np
import random
import h5py
import uuid
import torchvision.models.detection.retinanet
from sklearn.preprocessing import LabelEncoder


def plant():
    dir = '/home/neptun/PycharmProjects/dataset/dataset_plant'

    labels = os.listdir(dir)

    print(labels)

    list_labels = []
    list_uuid = []
    dict_numpy = {}

    for label in labels:
        path_lab_dir = os.path.join(dir, label)

        all_images = os.listdir(path_lab_dir)
        for name_img in all_images:
            image = Image.open(os.path.join(path_lab_dir, name_img))
            image = image.resize((224, 224))
            np_img = np.array(image)
            if np_img.shape[2] != 3:
                continue
            id = str(uuid.uuid4())

            dict_numpy[id] = np_img
            # dict_numpy[id] = np.moveaxis(np_img, 2, 0)
            list_labels.append(label)
            list_uuid.append(id)

    with h5py.File('data.hdf5', 'w') as f:
        for k, v in dict_numpy.items():
            f.create_dataset(k, data=v)

    np.save('label.npy', list_labels)
    np.save('id.npy', list_uuid)

def dota(train=False, val=False):
    if train == True:
        with h5py.File('train_data.hdf5', 'w') as f5:

            dir = '/home/neptun/PycharmProjects/datasets/dota/train'
            dir_labels = os.path.join(dir, 'DOTA-v2.0_train_hbb')
            first_ids_txt = sorted(os.listdir(dir_labels))
            rows = {}
            labels = []
            list_ids = []
            for first_id in first_ids_txt:
                id = first_id.split('.')[0]
                with open(os.path.join(dir_labels, first_id)) as f:
                    target = []
                    list_ids.append(id)
                    for row in f.readlines():
                        arr = row.split(' ')
                        box = np.array(arr[:8]).astype(np.float64)
                        label = arr[8]
                        labels.append(label)
                        target.append((box, label))
                    rows[id] = target
            # labels = set(labels)
            le = LabelEncoder()
            le.fit(labels)
            # print(len(rows))
            # print(labels)

            dirs_img = ['part1', 'part2', 'part3', 'part4', 'part5', 'part6']
            images_raw = {}
            for diri in dirs_img:
                path = os.path.join(dir, diri)
                images_in_dir = os.listdir(path)
                for x in images_in_dir:
                    id = x.split('.')[0]
                    images_raw[id] = os.path.join(path, x)

            # split image
            delta = 224
            dataset_dota = {}
            out_id = []
            out_labels = []
            for id in list_ids:
                print(id)
                try:
                    img = Image.open(images_raw[id])
                    kh, kw = (round(img.size[0]/delta, 1), round(img.size[1]/delta, 1))
                    h = img.size[0]
                    w = img.size[1]
                except:
                    continue
                img = np.array(img)
                if len(img.shape) == 2:
                    img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
                if len(img.shape) != 3:
                    continue
                if img.shape[2] != 3:
                    continue
                for i in range(2 * int(kw)):
                    for j in range(2 * int(kh)):
                        new_id = '_'.join([str(x) for x in [id, i, j, 2 * int(kw), 2 * int(kh)]])
                        x1 = int(i * delta / 2)
                        x2 = int((i+2)*delta / 2)
                        y1 = int(j * delta / 2)
                        y2 = int((j+2)*delta / 2)
                        img_loc = img[x1: x2, y1: y2, :]
                        if img_loc.shape[0] != delta or img_loc.shape[1] != delta:
                            if img_loc.shape[0] != delta:
                                x1 = w - delta
                                x2 = w
                            if img_loc.shape[1] != delta:
                                y1 = h - delta
                                y2 = h
                            img_loc = img[x1: x2, y1: y2, :]
                        targ_old = rows[id]
                        targ_box = []
                        targ_labl = []
                        for box, label in targ_old:
                            if x1 < box[1] < x2 and y1 < box[0] < y2 and \
                               x1 < box[3] < x2 and y1 < box[2] < y2 and \
                               x1 < box[5] < x2 and y1 < box[4] < y2 and \
                               x1 < box[7] < x2 and y1 < box[6] < y2:
                                bbb = box - np.array([y1, x1, y1, x1, y1, x1, y1, x1])
                                targ_box.append(np.array([bbb[0], bbb[1], bbb[4], bbb[5]]))
                                targ_labl.append(le.transform([label])[0])

                        # dataset_dota[new_id] = {'img': img_loc, 'target': targ_new}
                        # out_labels.append(targ_new)
                        if len(targ_labl) > 0:
                            out_id.append(new_id)
                            f5.create_dataset('img_'+new_id, data=img_loc)
                            f5.create_dataset('box_'+new_id, data=targ_box)
                            f5.create_dataset('label_'+new_id, data=targ_labl)

        # np.save('label.npy', out_labels)
        np.save('train_id.npy', out_id)
        print(len(out_id))
    if val == True:
        with h5py.File('val_data.hdf5', 'w') as f5:

            dir = '/home/neptun/PycharmProjects/datasets/dota/val'
            dir_labels = os.path.join(dir, 'DOTA-v2.0_val_hbb')
            first_ids_txt = os.listdir(dir_labels)
            rows = {}
            labels = []
            list_ids = []
            for first_id in first_ids_txt:
                id = first_id.split('.')[0]
                with open(os.path.join(dir_labels, first_id)) as f:
                    target = []
                    list_ids.append(id)
                    for row in f.readlines():
                        arr = row.split(' ')
                        box = np.array(arr[:8]).astype(np.float64)
                        label = arr[8]
                        labels.append(label)
                        target.append((box, label))
                    rows[id] = target
            # labels = set(labels)
            print(len(set(labels)))
            le = LabelEncoder()
            le.fit(labels)
            # code_label = le.transform(labels)

            # print(len(rows))
            # print(labels)

            dirs_img = ['part1', 'part2', ]
            images_raw = {}
            for diri in dirs_img:
                path = os.path.join(dir, diri)
                images_in_dir = os.listdir(path)
                for x in images_in_dir:
                    id = x.split('.')[0]
                    images_raw[id] = os.path.join(path, x)

            # split image
            delta = 224
            dataset_dota = {}
            out_id = []
            out_labels = []
            for id in list_ids:
                print(id)
                try:
                    img = Image.open(images_raw[id])
                    kh, kw = (round(img.size[0]/delta, 1), round(img.size[1]/delta, 1))
                    h = img.size[0]
                    w = img.size[1]
                except:
                    continue
                img = np.array(img)
                if len(img.shape) == 2:
                    img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
                if len(img.shape) != 3:
                    continue
                if img.shape[2] != 3:
                    continue
                for i in range(2 * int(kw)):
                    for j in range(2 * int(kh)):
                        new_id = '_'.join([str(x) for x in [id, i, j, 2 * int(kw), 2 * int(kh)]])
                        x1 = int(i * delta / 2)
                        x2 = int((i+2)*delta / 2)
                        y1 = int(j * delta / 2)
                        y2 = int((j+2)*delta / 2)
                        img_loc = img[x1: x2, y1: y2, :]
                        if img_loc.shape[0] != delta or img_loc.shape[1] != delta:
                            if img_loc.shape[0] != delta:
                                x1 = w - delta
                                x2 = w
                            if img_loc.shape[1] != delta:
                                y1 = h - delta
                                y2 = h
                            img_loc = img[x1: x2, y1: y2, :]
                        targ_old = rows[id]
                        targ_box = []
                        targ_labl = []
                        for box, label in targ_old:
                            if x1 < box[1] < x2 and y1 < box[0] < y2 and \
                               x1 < box[3] < x2 and y1 < box[2] < y2 and \
                               x1 < box[5] < x2 and y1 < box[4] < y2 and \
                               x1 < box[7] < x2 and y1 < box[6] < y2:
                                bbb = box - np.array([y1, x1, y1, x1, y1, x1, y1, x1])
                                targ_box.append(np.array([bbb[0], bbb[1], bbb[4], bbb[5]]))
                                targ_labl.append(le.transform([label])[0])

                        # dataset_dota[new_id] = {'img': img_loc, 'target': targ_new}
                        # out_labels.append(targ_new)
                        if len(targ_labl) > 0:
                            out_id.append(new_id)
                            f5.create_dataset('img_'+new_id, data=img_loc)
                            f5.create_dataset('box_'+new_id, data=targ_box)
                            f5.create_dataset('label_'+new_id, data=targ_labl)

        # np.save('label.npy', out_labels)
        np.save('val_id.npy', out_id)
        print(len(out_id))


if __name__ == '__main__':
    dota(train=True)