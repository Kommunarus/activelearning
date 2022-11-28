import yaml
from autoencoder.PyTorch_VAE.models import *
from autoencoder.PyTorch_VAE.experiment import VAEXperiment
from pytorch_lightning.utilities.seed import seed_everything
from autoencoder.PyTorch_VAE.dataset import VAEDataset
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, MeanShift, estimate_bandwidth, AgglomerativeClustering
import collections
import random
from PIL import Image
import os
from sklearn.svm import OneClassSVM
from sklearn.manifold import TSNE
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

path_yaml_celeba = '/home/neptun/PycharmProjects/activelearning/models/vae_celeba.yaml'
path_yaml_plants = '/home/neptun/PycharmProjects/activelearning/models/vae_plants.yaml'
path_check = '/home/neptun/PycharmProjects/activelearning/models/last.ckpt'

class Feature_vae:
    def __init__(self, device):
        model = vae_models[config['model_params']['name']](**config['model_params'])
        experiment = VAEXperiment(model,
                                  config['exp_params'])
        experiment.load_from_checkpoint(path_check, vae_model=model, params=config['exp_params'])

        experiment.model.eval().to(device)
        self.model = experiment.model

    def predict(self, x):
        return self.model(x)

    def encode(self, x):
        return self.model.encode(x)

    def loss_function(self, *args, **kwargs):
        return self.model.loss_function(*args, **kwargs)


with open(path_yaml_celeba, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)
config['exp_params']['M_N'] = config['exp_params']['kld_weight']
seed_everything(config['exp_params']['manual_seed'], True)

N = 3000
config['data_params']['train_batch_size'] = 1
vae = Feature_vae('cuda:0')
data_celeba_man = VAEDataset(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0,
                         limit=N, filter_label=0)
data_celeba_man.setup()

f = []
indx = []
train_dataset = data_celeba_man.train_dataloader()
for x, idx in train_dataset:
    args = vae.encode(x.to('cuda:0'))
    f.append(args[0].tolist()[0])
    indx.append(idx.tolist()[0])

X = np.array(f)

data_celeba_woman = VAEDataset(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0,
                         limit=N, filter_label=1)
data_celeba_woman.setup()

f_w = []
indx_w = []
train_dataset_w = data_celeba_woman.train_dataloader()
for x, idx in train_dataset_w:
    args = vae.encode(x.to('cuda:0'))
    f_w.append(args[0].tolist()[0])
    indx_w.append(idx.tolist()[0])

Y = np.array(f_w)

XY = np.concatenate([X, Y], 0)

# X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=500).fit_transform(XY)
#
# # ax = plt.axes(projection ="3d")
# # ax.scatter3D(X_embedded[:N, 0], X_embedded[:N, 1], X_embedded[:N, 2], c='r', alpha=0.3)
# # ax.scatter3D(X_embedded[N:, 0], X_embedded[N:, 1], X_embedded[N:, 2], c='g', alpha=0.3)
# plt.scatter(X_embedded[:N, 0], X_embedded[:N, 1], c='r', alpha=0.3)
# plt.show()
# plt.scatter(X_embedded[N:, 0], X_embedded[N:, 1], c='g', alpha=0.3)
# plt.show()
# plt.scatter(X_embedded[:N, 0], X_embedded[:N, 1], c='r', alpha=0.3)
# plt.scatter(X_embedded[N:, 0], X_embedded[N:, 1], c='g', alpha=0.3)
# plt.show()
# one_class_svm = OneClassSVM(nu=0.1, kernel='rbf', degree=6).fit(X)
# out_man = one_class_svm.predict(X)
# count = collections.Counter(out_man)
# print(count)
# lof = LocalOutlierFactor(n_neighbors=40)
# lof = IsolationForest()
# out_man = lof.fit_predict(X)
# count = collections.Counter(out_man)
# print(count)

# out_woman = lof.predict(Y)
# count = collections.Counter(out_woman)
# print(count)

# with open(path_yaml_plants, 'r') as file:
#     try:
#         config = yaml.safe_load(file)
#     except yaml.YAMLError as exc:
#         print(exc)
# config['exp_params']['M_N'] = config['exp_params']['kld_weight']
# config['data_params']['train_batch_size'] = 1
# data_plants = VAEDataset(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0,
#                          limit=-1, filter_label=None)
# data_plants.setup()
#
# f_p = []
# indx_p = []
# train_dataset_p = data_plants.train_dataloader()
# for x, idx in train_dataset_p:
#     args = vae.encode(x.to('cuda:0'))
#     f_p.append(args[0].tolist()[0])
#     indx_p.append(idx.tolist()[0])
#
# Z = np.array(f_p)
#
# out_plants = one_class_svm.predict(Z)
# count = collections.Counter(out_plants)
# print(count)



# method = DBSCAN(eps=14, min_samples=10)
# clusterization = method.fit(X)
# labels = clusterization.labels_
#
# count = collections.Counter(labels)
# print(count)
# N = 5
# for k in [-1, 0]:
#     indx = np.where(clusterization.labels_ == k)[0]
#     if len(indx) > N**2:
#         indx = random.sample(list(indx), N**2)
#
#     out = [train_dataset.dataset.data[i] for i in indx]
#     fig, axs = plt.subplots(N, N)
#     m = -1
#     for i in range(N):
#         for j in range(N):
#             m += 1
#             if m+1 > len(indx):
#                 continue
#             file = out[m]
#             img = Image.open(file)
#             axs[i, j].imshow(img)
#             axs[i, j].axis('off')
#     fig.suptitle(k)
#     plt.show()
#
