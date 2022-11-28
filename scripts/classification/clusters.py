import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from random import shuffle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, MeanShift, estimate_bandwidth, AgglomerativeClustering

# from unit import Dataset_plants_free, read_dirs_dataset
# from train_plants import Feature

class Clusters():
    def __init__(self, dataset, num_cluster):
        self.dataset = dataset
        self.num_cluster = num_cluster


    def make_cluster(self):
        list_featch = []
        list_id = []
        for batch in self.dataset:
            list_featch.append(np.squeeze(batch[0].detach().cpu().numpy()))
            list_id.append(batch[2])

        X = np.array(list_featch)
        # pca = PCA(n_components=0.8)
        # pcaX = pca.fit_transform(X)

        method = KMeans(n_clusters=self.num_cluster, max_iter=300, algorithm='lloyd')
        # method = DBSCAN(eps=3, min_samples=10)
        # bandwidth = estimate_bandwidth(X, quantile=0.3, n_samples=100)
        # method = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        # method = AgglomerativeClustering(linkage="average", n_clusters=self.num_cluster)

        clusterization = method.fit(X)
        self.labels = clusterization.labels_
        # self.cluster_centers_ = kmeans.cluster_centers_
        self.list_id = list_id
        self.x = X

    def get_random_members(self, number=1):
        out = []
        n_cl = list(set(self.labels))
        n_clusters = len(n_cl)
        if n_clusters > number:
            n_in_cl_1 = 0
            n_in_cl_2 = number
        else:
            n_in_cl_1 = number // n_clusters
            n_in_cl_2 = number % n_clusters

        if n_in_cl_1 != 0:
            for la in n_cl:
                indx = np.where(self.labels == la)[0]
                if len(indx) > n_in_cl_1:
                    indx = random.sample(list(indx), n_in_cl_1)
                out = out + [self.list_id[i] for i in indx]

        if n_in_cl_2 != 0:
            work_cluster = random.sample(n_cl, n_in_cl_2)
            for la in work_cluster:
                indx = np.where(self.labels == la)[0]
                indx = random.sample(list(indx), 1)
                out = out + [self.list_id[i] for i in indx]

        return out



if __name__ == '__main__':
    pass
    # path_to_dataset_train = '/home/neptun/PycharmProjects/datasets/flower_data/train'
    # path_to_dataset_val = '/home/neptun/PycharmProjects/datasets/flower_data/valid'
    # path_to_dataset_numpy = '/home/neptun/PycharmProjects/datasets/flower_data/numpy'
    #
    # device = 'cuda:0'
    #
    # model_feacher = Feature(device)
    #
    # dict_id, num_labels = read_dirs_dataset(path_to_dataset_train)
    #
    # all_items = [x for x in dict_id.keys()]
    #
    #
    # dataset = Dataset_plants_free(all_items, path_to_dataset_train, model_feacher, device,
    #                                     path_to_dataset_numpy)
    #
    # train_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    #
    # clu = Clusters(train_dataloader)
    # clu.make_cluster()
    # out = clu.get_random_members(3)
    # print(out)