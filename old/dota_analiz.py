import numpy as np
import os
import h5py
import collections


dir_to_dataset = '/home/neptun/PycharmProjects/dataset/dota'

id_all = np.load(os.path.join(dir_to_dataset, 'train_id.npy')).tolist()
archives = h5py.File(os.path.join(dir_to_dataset, 'train_data.hdf5'), 'r')
cd = []

for id in id_all:
    all_labels = archives.get('label_' + id)[:].tolist()
    cd = cd + all_labels

counter = collections.Counter(cd)
print(counter)