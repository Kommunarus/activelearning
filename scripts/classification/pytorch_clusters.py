import torch
import torch.nn.functional as F
from random import shuffle
import numpy as np

class CosineClusters():

    def __init__(self, model_feacher, device, num_clusters=100):

        self.clusters = []  # clusters for unsupervised and lightly supervised sampling
        self.item_cluster = {}  # each item's cluster by the id of the item

        # Create initial clusters
        for i in range(0, num_clusters):
            self.clusters.append(Cluster(model_feacher, device))

    def add_random_training_items(self, dataloader):
        """ Adds items randomly to clusters
        """

        cur_index = 0
        for batch in dataloader:
            self.clusters[cur_index].add_to_cluster(batch)
            n = len(batch[2])
            for i in range(n):
                fileid = batch[2][i]
                self.item_cluster[fileid] = self.clusters[cur_index]

            cur_index += 1
            if cur_index >= len(self.clusters):
                cur_index = 0

    def add_items_to_best_cluster(self, dataloader):
        """ Adds multiple items to best clusters

        """
        added = 0
        for batch in dataloader:
            new = self.add_item_to_best_cluster(batch)
            added += sum(new)

        return added

    def get_best_cluster(self, item):
        """ Finds the best cluster for this item

            returns the cluster and the score
        """
        best_cluster = None
        best_fit = float("-inf")

        for cluster in self.clusters:
            fit = cluster.cosine_similary_feat(item)
            if fit > best_fit:
                best_fit = fit
                best_cluster = cluster

        return [best_cluster, best_fit]

    def add_item_to_best_cluster(self, batch):
        best_cluster = None
        best_fit = float("-inf")
        previous_cluster = None

        out = []

        # Remove from current cluster so it isn't contributing to own match
        fileids = batch[2]
        for i, fileid in enumerate(fileids):
            old_featch = None
            if fileid in self.item_cluster:
                previous_cluster = self.item_cluster[fileid]
                old_featch = previous_cluster.remove_from_cluster(fileid)

            for cluster in self.clusters:
                fit = cluster.cosine_similary_feat(old_featch)
                if fit > best_fit:
                    best_fit = fit
                    best_cluster = cluster

            best_cluster.add_to_cluster([batch[0][i], None, [fileid, ], old_featch])
            self.item_cluster[fileid] = best_cluster

            if best_cluster == previous_cluster:
                out.append(0)
            else:
                out.append(1)

        return out

    def get_items_cluster(self, item):
        fileid = item[0]

        if fileid in self.item_cluster:
            return self.item_cluster[fileid]
        else:
            return None

    def get_centroids(self):
        centroids = []
        for cluster in self.clusters:
            centr = cluster.get_centroid()
            if not (centr is None):
                centroids.append(centr)

        return centroids

    def get_outliers(self):
        outliers = []
        for cluster in self.clusters:
            outlier = cluster.get_outlier()
            if not (outlier is None):
                outliers.append(outlier)

        return outliers

    def get_randoms(self, number_per_cluster=1, verbose=False):
        randoms = []
        for cluster in self.clusters:
            rand = cluster.get_random_members(number_per_cluster, verbose)
            if not (rand is None):
                randoms += rand

        return randoms

    def shape(self):
        lengths = []
        for cluster in self.clusters:
            lengths.append(cluster.size())

        return str(lengths)


class Cluster():
    """Represents on cluster for unsupervised or lightly supervised clustering

    """

    # feature_idx = {}  # the index of each feature as class variable to be constant

    def __init__(self, model_feacher, device):
        self.feature = {}  # dict of items by item ids in this cluster
        self.feature_vector = np.zeros((1280, ))  # feature vector for this cluster
        self.model_feacher = model_feacher
        self.device = device

    def add_to_cluster(self, batch):
        fileid = batch[2]
        features = batch[0].to(self.device)

        if len(batch) == 4:
            features = batch[3]
        else:
            # features = self.model_feacher.predict(img)
            features = features.detach().cpu().numpy()

        n = self.size()
        m = len(batch[2])
        old_feature_vector = self.feature_vector


        if m > 1:
            for i in range(m):
                self.feature[fileid[i]] = np.squeeze(features[i])
            self.feature_vector = m/(n+m) * np.mean(features, 0) + n/(n+m) * old_feature_vector
        else:
            self.feature[fileid[0]] = features
            self.feature_vector = m/(n+m) * features + n/(n+m) * old_feature_vector


    def remove_from_cluster(self, fileid):
        """ Removes if exists in the cluster

        """
        n = self.size()
        exists = self.feature.pop(fileid, False)
        if isinstance(exists, np.ndarray):
            if n > 1:
                self.feature_vector = n/(n-1) * self.feature_vector - exists/(n-1)
            else:
                self.feature_vector = np.zeros((1280,))
        return exists

    def cosine_similary(self, batch):
        imgs = batch[0].to(self.device)
        item_tensor = self.model_feacher.predict(imgs)
        cluster_tensor = torch.FloatTensor(self.feature_vector).to(self.device)
        similarity = F.cosine_similarity(item_tensor, cluster_tensor, 1)
        return similarity.detach().cpu().numpy()  # item() converts tensor value to float

    def cosine_similary_feat(self, feature):
        item_tensor = torch.FloatTensor(feature)
        cluster_tensor = torch.FloatTensor(self.feature_vector)
        similarity = F.cosine_similarity(item_tensor, cluster_tensor, 0)
        return similarity.item()  # item() converts tensor value to float

    def size(self):
        return len(self.feature.keys())

    def get_centroid(self):
        if len(self.feature) == 0:
            return None

        best_item = None
        best_fit = float("-inf")

        for fileid in self.feature.keys():
            feat = self.feature[fileid]
            similarity = self.cosine_similary_feat(feat)
            if similarity > best_fit:
                best_fit = similarity
                best_item = fileid

        return best_item, best_fit, 'centroid'

    def get_outlier(self):
        if len(self.feature) == 0:
            return None

        best_item = None
        biggest_outlier = float("inf")

        for fileid in self.feature.keys():
            feat = self.feature[fileid]
            similarity = self.cosine_similary_feat(feat)
            if similarity < biggest_outlier:
                biggest_outlier = similarity
                best_item = fileid
        return best_item, 1 - biggest_outlier, 'outlier'

    def get_random_members(self, number=1, verbose=False):
        if len(self.feature) == 0:
            return None

        keys = list(self.feature.keys())
        shuffle(keys)

        randoms = []
        for i in range(0, number):
            if i < len(keys):
                fileid = keys[i]
                feat = self.feature[fileid]
                sim = self.cosine_similary_feat(feat)

                randoms.append((fileid, sim, 'random'))

        return randoms





