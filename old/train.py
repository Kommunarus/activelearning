import torch
import torch.nn as nn

from torchvision import transforms
import random
from sklearn.metrics import f1_score
import yaml
from torch.utils.data import DataLoader
from unit import NeuralNetwork, Dataset
from algorithm import least_confidence
from pytorch_clusters import Cluster

with open('setting.yaml') as f:
    templates = yaml.safe_load(f)

random.seed(templates['randomseed'])
device = "cuda:1" if torch.cuda.is_available() else "cpu"
data_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])



def train_model(first, step=0, limit=-1):

    dataset_train = Dataset(transform=data_transform, first=first, val=False, step=step, limit=limit, al='train_model')
    print('lenght dataset is {}'.format(len(dataset_train)))
    train_dataloader = DataLoader(dataset_train, batch_size=32, shuffle=True)
    dataset_val = Dataset(transform=data_transform, val=True)
    val_dataloader = DataLoader(dataset_val, batch_size=32, shuffle=True)

    loss_func = nn.CrossEntropyLoss()
    model = NeuralNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # train_features, train_labels = next(iter(train_dataloader))
    best_score = 0
    for ep in range(1, templates['n_epoch'] + 1):
        sumloss = 0
        y_true = []
        y_pred = []
        model.train()
        for batch in train_dataloader:
            imgs = batch[0].to(device)
            labs = batch[1].to(device)
            out, prob = model(imgs)

            loss = loss_func(out, labs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sumloss += loss.item()

            y_true = y_true + labs.tolist()
            pred = torch.argmax(prob, 1).tolist()
            y_pred = y_pred + pred

        acc = f1_score(y_true, y_pred)
        print('train ep {}, loss {:.3f}, f1 {:.3f}'.format(ep, sumloss/len(train_dataloader), acc))

        model.eval()
        sumloss = 0
        y_true = []
        y_pred = []
        for batch in val_dataloader:
            imgs = batch[0].to(device)
            labs = batch[1].to(device)
            out, prob = model(imgs)

            loss = loss_func(out, labs)
            sumloss += loss.item()

            y_true = y_true + labs.tolist()
            pred = torch.argmax(prob, 1).tolist()
            y_pred = y_pred + pred

        acc = f1_score(y_true, y_pred)
        print('val ep {}, loss {:.3f}, f1 {:.3f}'.format(ep, sumloss/len(val_dataloader), acc))
        if best_score < acc:
            print('save model')
            best_score = acc
            torch.save(model.state_dict(), f'model_weights_{step}.pth')

def sampling_uncertainty(step):
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load(f'model_weights_{step}.pth', map_location=device))
    model.eval()
    dataset_train = Dataset(transform=data_transform, step=step, al='find_score')
    train_dataloader = DataLoader(dataset_train, batch_size=32, shuffle=False)
    indexs = []
    values = []
    for i, batch in enumerate(train_dataloader):
        imgs = batch[0].to(device)
        out, prob = model(imgs)
        confidence = least_confidence(prob)

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


if __name__ == '__main__':
    pass
    # train_model(True, limit=-1, step=0)
    # sampling_uncertainty(step=0)
    # print('AL1')
    # train_model(False, limit=-1, step=1)
    # sampling_uncertainty(step=1)
    # train_model(False, limit=-1, step=2)
    # sampling_uncertainty(step=2)
    # train_model(False, limit=-1, step=3)
    # sampling_uncertainty(step=3)
    # train_model(False, limit=-1, step=4)
    # sampling_uncertainty(step=4)
    # train_model(False, limit=-1, step=5)
    # sampling_uncertainty(step=5)
    # train_model(False, limit=-1, step=6)
    # print('AL2')
    # train_model(False, limit=-1, step=1)
    # print('AL3')
    # train_model(False, limit=-1, step=1)
    # print('R1')
    # train_model(False, limit=12000, step=10)
    # print('R2')
    # train_model(False, limit=2000, step=1)
    # print('R3')
    # train_model(False, limit=2000, step=1)