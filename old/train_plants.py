import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from torchvision import transforms
# from torchvision.models import efficientnet_b0
import random
from random import shuffle

from sklearn.metrics import f1_score
import yaml
from torch.utils.data import DataLoader
from unit import NeuralNetwork, NeuralNetwork_transfer, Dataset_plant, Dataset_plants_free, prepare_items, good_func
from unit import Dataset_transfer_learning
from algorithm import least_confidence, margin_confidence, ratio_confidence, entropy_based
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from pytorch_clusters import Cluster, CosineClusters
from ejection import get_model_outliers
import copy
import collections
import math

# https://github.com/rmunro/pytorch_active_learning

with open('setting.yaml') as f:
    templates = yaml.safe_load(f)

random.seed(templates['randomseed'])
device = "cuda:0" if torch.cuda.is_available() else "cpu"

class Feature():
    def __init__(self):
        efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)

        fc = nn.Sequential(
              nn.AdaptiveAvgPool2d(output_size=1),
              nn.Flatten()
        )
        efficientnet.classifier = fc
        efficientnet.eval().to(device)
        self.efficientnet = efficientnet

    def predict(self, x):
        return self.efficientnet(x)


def train_model(model_feacher, items_train, items_val):

    ds0 = Dataset_plants_free(items_train)
    ds_val = Dataset_plants_free(items_val)

    train_dataloader = DataLoader(ds0, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(ds_val, batch_size=32, shuffle=True)

    loss_func = nn.CrossEntropyLoss()
    model = NeuralNetwork(num_classes=39).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # train_features, train_labels = next(iter(train_dataloader))
    best_score = 0
    train_acc = 0
    model_best = None
    for ep in range(1, templates['n_epoch'] + 1):
        sumloss = 0
        y_true = []
        y_pred = []
        model.train()
        for batch in train_dataloader:
            imgs = batch[0].to(device)
            labs = batch[1].to(device)

            fea = model_feacher.predict(imgs)
            _, out, prob = model(fea)

            loss = loss_func(out, labs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sumloss += loss.item()

            y_true = y_true + labs.tolist()
            pred = torch.argmax(prob, 1).tolist()
            y_pred = y_pred + pred

        train_acc = f1_score(y_true, y_pred, average='macro')

        model.eval()
        sumloss = 0
        y_true = []
        y_pred = []
        for batch in val_dataloader:
            imgs = batch[0].to(device)
            labs = batch[1].to(device)

            fea = model_feacher.predict(imgs)
            _, out, prob = model(fea)

            loss = loss_func(out, labs)
            sumloss += loss.item()

            y_true = y_true + labs.tolist()
            pred = torch.argmax(prob, 1).tolist()
            y_pred = y_pred + pred

        acc = f1_score(y_true, y_pred, average='macro')
        if best_score < acc:
            # print('train ep {}, f1 {:.3f}'.format(ep, train_acc))
            # print('val ep {}, f1 {:.3f}'.format(ep, acc))
            # print('save model')
            best_score = acc
            model_best = copy.deepcopy(model)
            # torch.save(model.state_dict(), f'./models/model_weights_{step}.pth')
    print('val best f1 {:.3f}, train f1 {:.3f}'.format(best_score, train_acc))
    return model_best

def sampling_uncertainty(model, model_feacher, unlabeled_data, method='margin'):
    model.eval()
    dataset_train = Dataset_plants_free(unlabeled_data)
    train_dataloader = DataLoader(dataset_train, batch_size=32, shuffle=False)
    indexs = []
    values = []
    for i, batch in enumerate(train_dataloader):
        imgs = batch[0].to(device)
        features = model_feacher.predict(imgs)
        out, prob = model(features)

        if method == 'least':
            confidence = least_confidence(prob)
        elif method == 'margin':
            confidence = margin_confidence(prob)
        elif method == 'ratio':
            confidence = ratio_confidence(prob)
        else:
            confidence = entropy_based(prob, 20)

        indexs += [x for x in batch[2]]
        values += confidence
        # print('{} of {}'.format(i, len(train_dataloader)))

    temp = sorted(values)[-templates['num_for_al']:]
    out_name = []
    out_value = []
    for ell in temp:
        indx = values.index(ell)
        out_value.append(ell)
        out_name.append(indexs[indx])
    return out_name

    # with open(f'step{step+1}.txt', 'w') as f:
    #     f.write('; '.join([str(x) for x in out_name]))
    #     f.write('\n')
    #     f.write('; '.join([str(x) for x in out_value]))

def outliers(step, model_feacher):
    model = NeuralNetwork().to(device)
    if step != 0:
        model.load_state_dict(torch.load(f'./models/model_weights_{step}.pth', map_location=device))
    model.eval()
    unlabeled_data = Dataset_plant(transform=data_transform, step=step, al='-', type='train', limit=10000)
    unlabeled_data_dl = DataLoader(unlabeled_data, batch_size=1, shuffle=False)

    dataset_test = Dataset_plant(transform=data_transform, type='test')
    val_dl = DataLoader(dataset_test, batch_size=1, shuffle=True)

    out_name = get_model_outliers(model, unlabeled_data_dl, val_dl, model_feacher, device,
                                  number=templates['num_for_al'])
    with open(f'old/models/step{step + 1}.txt', 'w') as f:
        f.write('; '.join([str(x) for x in out_name]))

def get_representative_samples(training_data, unlabeled_data, model_feacher, number=20):
    training_cluster = Cluster(model_feacher, device)
    for item in training_data:
        training_cluster.add_to_cluster(item)

    unlabeled_cluster = Cluster(model_feacher, device)
    for item in unlabeled_data:
        unlabeled_cluster.add_to_cluster(item)

    out = []
    for item in unlabeled_data:
        training_score = training_cluster.cosine_similary(item)
        unlabeled_score = unlabeled_cluster.cosine_similary(item)

        representativeness = unlabeled_score - training_score

        out = out + [(x, y, z, "representative") for x, y, z in zip(item[2], item[1], representativeness)]

    out.sort(reverse=True, key=lambda x: x[1])
    return out[:number:]

def get_adaptive_representative_samples(training_data, unlabeled_data, model_feacher, number=20):
    samples = []
    exclude = []
    for i in range(0, number):
        print("Epoch " + str(i))
        representative_item = get_representative_samples(training_data, unlabeled_data, model_feacher, 1)[0]
        samples.append(representative_item)
        exclude.append(representative_item[0])

        dataset_val = Dataset_plant(transform=data_transform, type='val', exclude=exclude)
        unlabeled_data = DataLoader(dataset_val, batch_size=32, shuffle=True)

        # unlabeled_data.remove(representative_item)

    return samples

def get_cluster_samples(unlabeled_data, num_clusters=5, max_epochs=5, num=3):

    cosine_clusters = CosineClusters(model_feacher, device, num_clusters)

    dataset_train = Dataset_plants_free(unlabeled_data)
    train_dataloader = DataLoader(dataset_train, batch_size=32, shuffle=False)
    cosine_clusters.add_random_training_items(train_dataloader)

    for i in range(0, max_epochs):
        # print("Epoch " + str(i))
        added = cosine_clusters.add_items_to_best_cluster(train_dataloader)
        if added == 0:
            break

    centroids = cosine_clusters.get_centroids()
    outliers = cosine_clusters.get_outliers()
    randoms = cosine_clusters.get_randoms(num)

    return centroids + outliers + randoms


def get_representative_cluster_samples(training_data, unlabeled_data, model_feacher, number=10, num_clusters=20, max_epochs=10,
                                       ):
    # if limit > 0:
    #     shuffle(training_data)
    #     training_data = training_data[:limit]
    #     shuffle(unlabeled_data)
    #     unlabeled_data = unlabeled_data[:limit]

    # Create clusters for training data

    dataset_train = Dataset_plants_free(training_data)
    train_dataloader = DataLoader(dataset_train, batch_size=32, shuffle=False)

    training_clusters = CosineClusters(model_feacher, device, num_clusters)
    training_clusters.add_random_training_items(train_dataloader)

    for i in range(0, max_epochs):
        print("Epoch " + str(i))
        added = training_clusters.add_items_to_best_cluster(train_dataloader)
        if added == 0:
            break

    # Create clusters for unlabeled data
    dataset_unlabeled = Dataset_plants_free(unlabeled_data)
    unlabeled_dataloader = DataLoader(dataset_unlabeled, batch_size=32, shuffle=False)

    unlabeled_clusters = CosineClusters(model_feacher, device, num_clusters)
    unlabeled_clusters.add_random_training_items(unlabeled_dataloader)

    for i in range(0, max_epochs):
        print("Epoch " + str(i))
        added = unlabeled_clusters.add_items_to_best_cluster(unlabeled_dataloader)
        if added == 0:
            break

    # get scores

    most_representative_items = []

    # for each cluster of unlabeled data
    for cluster in unlabeled_clusters.clusters:
        most_representative = None
        representativeness = float("-inf")

        # find the item in that cluster most like the unlabeled data
        item_keys = list(cluster.feature.keys())

        for key in item_keys:
            item = cluster.feature[key]

            _r, unlabeled_score = unlabeled_clusters.get_best_cluster(item)
            _, training_score = training_clusters.get_best_cluster(item)

            cluster_representativeness = unlabeled_score - training_score

            if cluster_representativeness > representativeness:
                representativeness = cluster_representativeness
                most_representative = [key, "representative_clusters", 0]

        if not (most_representative is None):
            most_representative[2] = representativeness
            most_representative_items.append(most_representative)

    most_representative_items.sort(reverse=True, key=lambda x: x[2])
    return most_representative_items[:number:]

def get_deep_active_transfer_learning_uncertainty_samples(model, unlabeled_data, validation_data, model_feacher,
                                                          new_val_samples=[], epochs=10, number=200):

    correct_predictions = []  # validation items predicted correctly
    incorrect_predictions = []  # validation items predicted incorrectly
    item_hidden_layers = {}  # hidden layer of each item, by id

    # 1 GET PREDICTIONS ON VALIDATION DATA FROM MODEL

    dataset_val = Dataset_plants_free(validation_data)
    val_dataloader = DataLoader(dataset_val, batch_size=16, shuffle=False)

    for batch in val_dataloader:

        ids = batch[2]
        imgs = batch[0]
        label = batch[1]

        feature_vector = model_feacher.predict(imgs.to(device))
        hidden, out, probs = model(feature_vector)

        for i in range(len(ids)):
            if ids[i] in new_val_samples:
                correct_predictions.append(ids[i])
            else:
                item_hidden_layers[ids[i]] = hidden[i].detach().cpu().numpy()
                maxprob = torch.max(probs, 1)
                if (label[i] == maxprob[1][i] and maxprob[0][i] > 0.5):
                    correct_predictions.append(ids[i])
                else:
                    incorrect_predictions.append(ids[i])

        # item.append(hidden) # the hidden layer will be the input to our new model

    # 2 BUILD A NEW MODEL TO PREDICT WHETHER VALIDATION ITEMS WERE CORRECT OR INCORRECT
    correct_model = NeuralNetwork_transfer().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(correct_model.parameters(), lr=1e-3)

    # print(correct_predictions)

    dataset_2 = Dataset_transfer_learning(item_hidden_layers, correct_predictions, incorrect_predictions)
    dataloader_2 = DataLoader(dataset_2, batch_size=128, shuffle=True)
    for epoch in range(epochs):
        sumloss = 0

        # train the final layers model
        for batch_mini in dataloader_2:
            feature_vec = batch_mini[0].to(device)
            label = batch_mini[1].to(device)
            correct_model.zero_grad()

            prob, _ = correct_model(feature_vec)

            # compute loss function, do backward pass, and update the gradient
            loss = loss_function(prob, label)
            loss.backward(retain_graph=True)
            optimizer.step()
            sumloss += loss.item()
        # print(sumloss)

    # 3 PREDICT WHETHER UNLABELED ITEMS ARE CORRECT

    deep_active_transfer_preds = []

    with torch.no_grad():
        dataset_unlabeled = Dataset_plants_free(unlabeled_data)
        unlabeled_dataloader = DataLoader(dataset_unlabeled, batch_size=32, shuffle=False)

        for batch in unlabeled_dataloader:
            imgs = batch[0]

            # get prediction from main model
            feature_vector = model_feacher.predict(imgs.to(device))
            hidden, logits, probs = model(feature_vector)

            # use hidden layer from main model as input to model predicting correct/errors
            logits, probs = correct_model(hidden)

            # get confidence that item is correctly labeled
            # prob_correct = 1 - math.exp(log_probs.data.tolist()[0][1])

            # if (label == "0"):
            #     prob_correct = 1 - prob_correct

            # item[3] = "predicted_error"
            # item[4] = 1 - prob_correct
            for id, score in zip(batch[2], probs[:,0]):
                s = [id, "predicted_error", score]
                deep_active_transfer_preds.append(s)

    deep_active_transfer_preds.sort(reverse=True, key=lambda x: x[2])

    return deep_active_transfer_preds[:number:]

def get_atlas_samples( model, unlabeled_data, validation_data, feature_method,
                      number=200):
    if (len(unlabeled_data) < number):
        raise Exception('More samples requested than the number of unlabeled items')

    atlas_samples = []  # all items sampled by atlas
    new_val_samples = []  # all items sampled by atlas

    print(number)
    while (len(atlas_samples) < number):
        samples = get_deep_active_transfer_learning_uncertainty_samples(model, unlabeled_data, validation_data,
                                                                             feature_method, new_val_samples,
                                                                             number=200)
        for item in samples:
            atlas_samples.append([item[0], 'atlas', 0])
            unlabeled_data.remove(item[0])

            new_val_samples.append(item[0])  # append so that it is in the next iteration

    return atlas_samples


if __name__ == '__main__':
    model_feacher = Feature()

    dict_id = good_func()

    all_items = [x for x in dict_id.keys()]
    labeled_data = prepare_items(limit=10000, del_labels=[1,5,15,20,27,35], train=True, seed=1)
    items_val = prepare_items(val=True)
    # model0 = train_model(model_feacher, labeled_data, items_val)
    # torch.save(model0.state_dict(), f'./models/model_weights_0.pth')
    model0 = NeuralNetwork(39).to(device)
    model0.load_state_dict(torch.load(f'./models/model_weights_0.pth', map_location=device))

    unlabeled_data = list(set(all_items) - set(labeled_data) - set(items_val))

    # add_to_label_items = sampling_uncertainty(model0, model_feacher, unlabeled_data, method='margin')
    # add2_to_label_items = get_cluster_samples(add_to_label_items, num_clusters=10, max_epochs=10, num=200)
    #
    # print('AL')
    # labeled_data2 = labeled_data + [x[0] for x in add2_to_label_items]
    # train_model(model_feacher, labeled_data2, items_val)
    #
    # print('RND')
    # labeled_data2 = labeled_data + random.sample(unlabeled_data, k=200)
    # train_model(model_feacher, labeled_data2, items_val)
    # add_to_label_items = get_deep_active_transfer_learning_uncertainty_samples(model0, unlabeled_data,
    #                                                                            items_val, model_feacher,
    #                                                                            epochs=50, number=2000)
    # add_to_label_items = get_representative_cluster_samples(labeled_data, unlabeled_data, model_feacher, max_epochs=10,
    #                                                         num_clusters=40, number=40)

    add_to_label_items = get_atlas_samples(model0, unlabeled_data, items_val, model_feacher, number=2000)
    print(collections.Counter([dict_id[x[0]][1] for x in add_to_label_items]))

    # for i in range(10):
    # #     outliers(i, model_feacher)
    # # #     # clustering(i, templates['num_for_al'], model_feacher)
    #     sampling_uncertainty(step=i, type='least')
    #     train_model(model_feacher, False, limit=-1, step=i+1)
    # # train_model(model_feacher, False, limit=(i+1) * templates['num_for_al'], step=1000)

