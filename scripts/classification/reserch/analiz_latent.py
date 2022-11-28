import yaml
from autoencoder.PyTorch_VAE.models import *
from autoencoder.PyTorch_VAE.experiment import VAEXperiment
from scripts.classification.unit import NeuralNetwork, NeuralNetwork_transfer, Dataset_plants_free, \
    prepare_items, read_dirs_dataset
from torch.utils.data import DataLoader
from scripts.classification.train_plants import Feature_vae
import matplotlib.pyplot as plt
from scripts.classification.clusters import Clusters
import random
from PIL import Image
import os

num_clusters = 10

device = "cuda:0" if torch.cuda.is_available() else "cpu"
path_to_dataset_train = '/home/neptun/PycharmProjects/datasets/flower_data (копия)/train'
path_to_dataset_val = '/home/neptun/PycharmProjects/datasets/flower_data (копия)/valid'
# path_to_dataset_numpy = '/home/neptun/PycharmProjects/datasets/flower_data (копия)/numpy'
path_to_dataset_numpy_vae = '/home/neptun/PycharmProjects/datasets/flower_data (копия)/numpy_vae'

model_feacher_vae = Feature_vae(device)

dict_id, num_labels = read_dirs_dataset(path_to_dataset_train)
all_items = [x for x in dict_id.keys()]
items_train = sorted(prepare_items(limit=len(all_items), train=True, seed=42, files=dict_id, test_size=0))

dict_id_val, num_labels_val = read_dirs_dataset(path_to_dataset_val)
items_val = prepare_items(limit=len(dict_id_val), train=True, seed=42, files=dict_id_val, test_size=0)

ds0 = Dataset_plants_free(items_train, path_to_dataset_train, model_feacher_vae, device, path_to_dataset_numpy_vae,
vae=True)
ds_val = Dataset_plants_free(items_val, path_to_dataset_val, model_feacher_vae, device, path_to_dataset_numpy_vae,
                             vae=True)

train_dataloader = DataLoader(ds0, batch_size=1, shuffle=True)
val_dataloader = DataLoader(ds_val, batch_size=1, shuffle=True)

clu = Clusters(val_dataloader, num_clusters)
clu.make_cluster()

num_clusters = len(list(set(clu.labels)))
clusters = list(set(clu.labels))
colors = np.random.rand(num_clusters)

x = []
y = []
c = []
for xy, cc in zip(clu.x, clu.labels):
    x.append(xy[0])
    y.append(xy[8])
    c.append(colors[cc])
plt.scatter(x, y, c=c)
plt.title('val')
# plt.show()

for k in clusters:
    indx = np.where(clu.labels == k)[0]
    if len(indx) > 100:
        indx = random.sample(list(indx), 100)

    out = [clu.list_id[i] for i in indx]
    fig, axs = plt.subplots(10, 10)
    m = -1
    for i in range(10):
        for j in range(10):
            m += 1
            if m+1 > len(indx):
                continue
            file = out[m][0]
            # img = Image.open(os.path.join(path_to_dataset_train, dict_id[file][0], file))
            img = Image.open(os.path.join(path_to_dataset_val, dict_id_val[file][0], file))
            axs[i, j].imshow(img)
            axs[i, j].axis('off')

    plt.show()

