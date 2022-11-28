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
from unit import NeuralNetwork, Dataset_cifar
from algorithm import least_confidence, margin_confidence, ratio_confidence, entropy_based
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from pytorch_clusters import Cluster, CosineClusters
from ejection import get_model_outliers

# https://github.com/rmunro/pytorch_active_learning

with open('setting.yaml') as f:
    templates = yaml.safe_load(f)

mean = [0.4914, 0.4822, 0.4465]
std = [0.2470, 0.2435, 0.2616]

random.seed(templates['randomseed'])
device = "cuda:0" if torch.cuda.is_available() else "cpu"
data_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean,  std=std)
])

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


def train_model(model_feacher, first, step=0, limit=-1):

    dataset_train = Dataset_cifar(transform=data_transform, first=first, type='train', step=step, limit=limit,
                            al='train_model')
    print('lenght dataset is {}'.format(len(dataset_train)))
    train_dataloader = DataLoader(dataset_train, batch_size=32, shuffle=True)
    dataset_val = Dataset_cifar(transform=data_transform, type='val')
    val_dataloader = DataLoader(dataset_val, batch_size=32, shuffle=True)

    loss_func = nn.CrossEntropyLoss()
    model = NeuralNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # train_features, train_labels = next(iter(train_dataloader))
    best_score = 0
    train_acc = 0
    for ep in range(1, templates['n_epoch'] + 1):
        sumloss = 0
        y_true = []
        y_pred = []
        model.train()
        for batch in train_dataloader:
            imgs = batch[0].to(device)
            labs = batch[1].to(device)

            fea = model_feacher.predict(imgs)
            out, prob = model(fea)

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
            out, prob = model(fea)

            loss = loss_func(out, labs)
            sumloss += loss.item()

            y_true = y_true + labs.tolist()
            pred = torch.argmax(prob, 1).tolist()
            y_pred = y_pred + pred

        acc = f1_score(y_true, y_pred, average='macro')
        if best_score < acc:
            print('train ep {}, f1 {:.3f}'.format(ep, train_acc))
            print('val ep {}, f1 {:.3f}'.format(ep, acc))
            # print('save model')
            best_score = acc
            torch.save(model.state_dict(), f'old/models/model_weights_{step}.pth')
    print('step {}, limit {}, val best f1 {:.3f}, train f1 {:.3f}'.format(step, limit, best_score, train_acc))

def sampling_uncertainty(step, type='margin'):
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load(f'./models/model_weights_{step}.pth', map_location=device))
    model.eval()
    dataset_train = Dataset_cifar(transform=data_transform, step=step, al='find_score')
    train_dataloader = DataLoader(dataset_train, batch_size=32, shuffle=False)
    indexs = []
    values = []
    for i, batch in enumerate(train_dataloader):
        imgs = batch[0].to(device)
        out, prob = model(imgs)
        if type == 'least':
            confidence = least_confidence(prob)
        elif type == 'margin':
            confidence = margin_confidence(prob)
        elif type == 'ratio':
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

    with open(f'step{step+1}.txt', 'w') as f:
        f.write('; '.join([str(x) for x in out_name]))
        f.write('\n')
        f.write('; '.join([str(x) for x in out_value]))

def outliers(step, model_feacher):
    model = NeuralNetwork().to(device)
    if step != 0:
        model.load_state_dict(torch.load(f'./models/model_weights_{step}.pth', map_location=device))
    model.eval()
    unlabeled_data = Dataset_cifar(transform=data_transform, step=step, al='-', type='train', limit=10000)
    unlabeled_data_dl = DataLoader(unlabeled_data, batch_size=1, shuffle=False)

    dataset_test = Dataset_cifar(transform=data_transform, type='test')
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

        out = out + [(x, y, "representative") for x, y in zip(item[2], representativeness)]

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

        dataset_val = Dataset_cifar(transform=data_transform, type='val', exclude=exclude)
        unlabeled_data = DataLoader(dataset_val, batch_size=32, shuffle=True)

        # unlabeled_data.remove(representative_item)

    return samples

def get_cluster_samples(unlabeled_data, num_clusters=5, max_epochs=5):

    cosine_clusters = CosineClusters(model_feacher, device, num_clusters)

    cosine_clusters.add_random_training_items(unlabeled_data)

    for i in range(0, max_epochs):
        print("Epoch " + str(i))
        added = cosine_clusters.add_items_to_best_cluster(unlabeled_data)
        if added == 0:
            break

    centroids = cosine_clusters.get_centroids()
    outliers = cosine_clusters.get_outliers()
    randoms = cosine_clusters.get_randoms(3)

    return centroids + outliers + randoms

if __name__ == '__main__':
    model_feacher = Feature()

    dataset_train = Dataset_cifar(transform=data_transform, type='train', step=0, limit=10000, al='train_model')
    train_dataloader = DataLoader(dataset_train, batch_size=32, shuffle=True)
    dataset_val = Dataset_cifar(transform=data_transform, type='val')
    val_dataloader = DataLoader(dataset_val, batch_size=32, shuffle=True)

    # out = get_representative_samples(train_dataloader, val_dataloader, model_feacher)
    out = get_adaptive_representative_samples(train_dataloader, val_dataloader, model_feacher, number=5)
    # out = get_cluster_samples(train_dataloader)
    print(out)

    # model_feacher = Feature()
    # # # train_model(model_feacher, True, limit=-1, step=0)
    # for i in range(10):
    #     outliers(i, model_feacher)
    # #     # clustering(i, templates['num_for_al'], model_feacher)
    # #     # sampling_uncertainty(step=i, type='least')
    #     train_model(model_feacher, False, limit=-1, step=i+1)
    # train_model(model_feacher, False, limit=(i+1) * templates['num_for_al'], step=1000)

