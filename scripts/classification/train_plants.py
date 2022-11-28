import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from scripts.classification.unit import NeuralNetwork, NeuralNetwork_transfer, Dataset_plants_free, \
    prepare_items, read_dirs_dataset
from scripts.classification.unit import Dataset_transfer_learning
from scripts.classification.algorithm import least_confidence, margin_confidence, ratio_confidence, entropy_based
from scripts.classification.pytorch_clusters import Cluster, CosineClusters
from scripts.classification.clusters import Clusters
import copy
import time
import random
import yaml
from autoencoder.PyTorch_VAE.models import *
from autoencoder.PyTorch_VAE.experiment import VAEXperiment


class Feature:
    def __init__(self, device):
        efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True,
                                      verbose=False)

        fc = nn.Sequential(
              nn.AdaptiveAvgPool2d(output_size=1),
              nn.Flatten()
        )
        efficientnet.classifier = fc
        efficientnet.eval().to(device)
        self.efficientnet = efficientnet

    def predict(self, x):
        return self.efficientnet(x)

class Feature_vae:
    def __init__(self, device):
        path_yaml = '/home/neptun/PycharmProjects/activelearning/models/vae_celeba.yaml'
        path_check = '/home/neptun/PycharmProjects/activelearning/models/last.ckpt'
        with open(path_yaml, 'r') as file:
            try:
                config = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)

        model = vae_models[config['model_params']['name']](**config['model_params'])
        experiment = VAEXperiment(model,
                                  config['exp_params'])
        experiment.load_from_checkpoint(path_check, vae_model=model, params=config['exp_params'])

        experiment.model.eval().to(device)
        self.model = experiment.model

    def predict(self, x):
        return self.model.encode(x)

def train_model(model_feacher, items_train, items_val, device, path_to_dir_train, path_to_dataset_val,
                path_to_dataset_numpy, num_labels):

    ds0 = Dataset_plants_free(items_train, path_to_dir_train, model_feacher, device, path_to_dataset_numpy)
    ds_val = Dataset_plants_free(items_val, path_to_dataset_val, model_feacher, device, path_to_dataset_numpy)

    train_dataloader = DataLoader(ds0, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(ds_val, batch_size=32, shuffle=True)

    loss_func = nn.CrossEntropyLoss()
    model = NeuralNetwork(num_classes=num_labels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # train_features, train_labels = next(iter(train_dataloader))
    best_score = 0
    model_best = None
    num_bad_f1 = 0
    for ep in range(1, 150):
        if num_bad_f1 > 5:
            break
        # print(ep, end=' ')
        # sumloss = 0
        y_true = []
        y_pred = []
        model.train()
        for batch in train_dataloader:
            fea = batch[0].to(device)
            labs = batch[1].to(device)

            # fea = model_feacher.predict(imgs)
            _, out, prob = model(fea)

            loss = loss_func(out, labs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # sumloss += loss.item()

            y_true = y_true + labs.tolist()
            pred = torch.argmax(prob, 1).tolist()
            y_pred = y_pred + pred

        train_acc = f1_score(y_true, y_pred, average='macro')

        model.eval()
        # sumloss = 0
        y_true = []
        y_pred = []
        for batch in val_dataloader:
            fea = batch[0].to(device)
            labs = batch[1].to(device)

            # fea = model_feacher.predict(imgs)
            _, out, prob = model(fea)

            # loss = loss_func(out, labs)
            # sumloss += loss.item()

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
            num_bad_f1 = 0
            # torch.save(model.state_dict(), f'./models/model_weights_{step}.pth')
        else:
            num_bad_f1 += 1

    # print('val best f1 {:.3f}, train f1 {:.3f}'.format(best_score, train_acc))
    return model_best, best_score

def sampling_uncertainty(model, device, model_feacher, unlabeled_data, path_to_dir, path_to_dataset_numpy,
                         method='margin', num_sample=100,
                         num_labels=0):
    model.eval()
    # print('find best samples')
    dataset_train = Dataset_plants_free(unlabeled_data, path_to_dir, model_feacher, device, path_to_dataset_numpy)
    train_dataloader = DataLoader(dataset_train, batch_size=32, shuffle=False)
    indexs = []
    values = []
    for i, batch in enumerate(train_dataloader):
        features = batch[0].to(device)
        # features = model_feacher.predict(imgs)
        _, out, prob = model(features)

        if method == 'least':
            confidence = least_confidence(prob, num_labels=num_labels)
        elif method == 'margin':
            confidence = margin_confidence(prob)
        elif method == 'ratio':
            confidence = ratio_confidence(prob)
        else:
            confidence = entropy_based(prob, num_labels=num_labels)

        indexs += [x for x in batch[2]]
        values += confidence
        # print('{} of {}'.format(i, len(train_dataloader)))


    out_dict = {k:v for k, v in zip(indexs, values)}
    a = sorted(out_dict.items(), key=lambda x: x[1])


    temp = a[-num_sample:]
    out_name = [k for k, v in temp]
    # out_value = []
    # for ell in temp:
    #     indx = values.index(ell)
    #     # out_value.append(ell)
    #     out_name.append(indexs[indx])
    return out_name

    # with open(f'step{step+1}.txt', 'w') as f:
    #     f.write('; '.join([str(x) for x in out_name]))
    #     f.write('\n')
    #     f.write('; '.join([str(x) for x in out_value]))

# def outliers(step, model_feacher):
#     model = NeuralNetwork().to(device)
#     if step != 0:
#         model.load_state_dict(torch.load(f'./models/model_weights_{step}.pth', map_location=device))
#     model.eval()
#     unlabeled_data = Dataset_plant(transform=data_transform, step=step, al='-', type='train', limit=10000)
#     unlabeled_data_dl = DataLoader(unlabeled_data, batch_size=1, shuffle=False)
#
#     dataset_test = Dataset_plant(transform=data_transform, type='test')
#     val_dl = DataLoader(dataset_test, batch_size=1, shuffle=True)
#
#     out_name = get_model_outliers(model, unlabeled_data_dl, val_dl, model_feacher, device,
#                                   number=templates['num_for_al'])
#     with open(f'old/models/step{step + 1}.txt', 'w') as f:
#         f.write('; '.join([str(x) for x in out_name]))

def get_representative_samples(device, labeled_data, unlabeled_data, model_feacher, path_to_dir, num_sample=20):
    training_cluster = Cluster(model_feacher, device)
    dataset_train = Dataset_plants_free(unlabeled_data, path_to_dir)
    labeled_dataloader = DataLoader(dataset_train, batch_size=32, shuffle=False)

    for item in labeled_dataloader:
        training_cluster.add_to_cluster(item)

    unlabeled_cluster = Cluster(model_feacher, device)
    dataset_test = Dataset_plants_free(unlabeled_data, path_to_dir)
    unlabeled_dataloader = DataLoader(dataset_test, batch_size=32, shuffle=False)
    for item in unlabeled_dataloader:
        unlabeled_cluster.add_to_cluster(item)

    out = []
    for item in unlabeled_dataloader:
        training_score = training_cluster.cosine_similary(item)
        unlabeled_score = unlabeled_cluster.cosine_similary(item)

        representativeness = unlabeled_score - training_score

        out = out + [(x, y, z, "representative") for x, y, z in zip(item[2], item[1], representativeness)]

    out.sort(reverse=True, key=lambda x: x[1])
    return [x[0] for x in out[:num_sample]]

def get_cluster_samples(device, unlabeled_data, model_feacher, path_to_dir, path_to_dataset_numpy,
                        num_clusters=5, max_epochs=5, num_sample=3):

    cosine_clusters = CosineClusters(model_feacher, device, num_clusters)

    dataset_train = Dataset_plants_free(unlabeled_data, path_to_dir, model_feacher, device, path_to_dataset_numpy)
    train_dataloader = DataLoader(dataset_train, batch_size=32, shuffle=False)
    cosine_clusters.add_random_training_items(train_dataloader)

    for i in range(0, max_epochs):
        # print("Epoch " + str(i))
        added = cosine_clusters.add_items_to_best_cluster(train_dataloader)
        if added == 0:
            break

    centroids = cosine_clusters.get_centroids()
    outliers = cosine_clusters.get_outliers()
    randoms = cosine_clusters.get_randoms(num_sample // num_clusters - 2)

    out = centroids + outliers + randoms
    out2 = [x[0] for x in out]
    out2 = list(set(out2))
    delta = num_sample - len(out2)
    while delta != 0:
        randoms2 = cosine_clusters.get_randoms(1)
        randoms2 = random.choices(randoms2, k=delta)
        out2 = out2 + [x[0] for x in randoms2]
        out2 = list(set(out2))
        delta = num_sample - len(out2)


    return out2

def get_new_cluster_samples(device, unlabeled_data, model_feacher, path_to_dir, path_to_dataset_numpy,
                        num_clusters=5, num_sample=3):


    dataset_train = Dataset_plants_free(unlabeled_data, path_to_dir, model_feacher, device, path_to_dataset_numpy,
                                        vae=True)
    train_dataloader = DataLoader(dataset_train, batch_size=1, shuffle=False)

    clu = Clusters(train_dataloader, num_clusters)
    clu.make_cluster()
    out = clu.get_random_members(num_sample)
    out = [x[0] for x in out]

    return out

def get_representative_cluster_samples(device, training_data, unlabeled_data, model_feacher,
                                       path_to_dir_train, path_to_dataset_val, path_to_dataset_numpy, num_sample=10,
                                       num_clusters=20,
                                       max_epochs=10,
                                       ):
    if num_sample > num_clusters:
        assert 'num_sample > num_clusters'
    dataset_train = Dataset_plants_free(training_data, path_to_dir_train, model_feacher, device, path_to_dataset_numpy)
    train_dataloader = DataLoader(dataset_train, batch_size=32, shuffle=False)

    training_clusters = CosineClusters(model_feacher, device, num_clusters)
    training_clusters.add_random_training_items(train_dataloader)

    for i in range(0, max_epochs):
        # print("Epoch " + str(i))
        added = training_clusters.add_items_to_best_cluster(train_dataloader)
        if added == 0:
            break

    # Create clusters for unlabeled data
    dataset_unlabeled = Dataset_plants_free(unlabeled_data, path_to_dir_train, model_feacher, device,
                                            path_to_dataset_numpy)
    unlabeled_dataloader = DataLoader(dataset_unlabeled, batch_size=32, shuffle=False)

    unlabeled_clusters = CosineClusters(model_feacher, device, num_clusters)
    unlabeled_clusters.add_random_training_items(unlabeled_dataloader)

    for i in range(0, max_epochs):
        # print("Epoch " + str(i))
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
    return [x[0] for x in most_representative_items[:num_sample]]

def get_deep_active_transfer_learning_uncertainty_samples(model, device, unlabeled_data, validation_data, model_feacher,
                                                          path_to_dir_train, path_to_dataset_val, path_to_dataset_numpy,
                                                          new_val_samples, epochs=10, select_per_epoch=200,
                                                          num_labels=1):

    correct_predictions = []  # validation items predicted correctly
    incorrect_predictions = []  # validation items predicted incorrectly
    item_hidden_layers = {}  # hidden layer of each item, by id

    # 1 GET PREDICTIONS ON VALIDATION DATA FROM MODEL

    dataset_val = Dataset_plants_free(validation_data + new_val_samples, path_to_dataset_val, model_feacher, device,
                                      path_to_dataset_numpy, path_to_dir_train)
    val_dataloader = DataLoader(dataset_val, batch_size=16, shuffle=True)

    for batch in val_dataloader:

        ids = batch[2]
        feature_vector = batch[0]
        label = batch[1]

        # feature_vector = model_feacher.predict(imgs.to(device))
        hidden, out, probs = model(feature_vector)
        maxprob = torch.max(probs, 1)

        for i in range(len(ids)):
            item_hidden_layers[ids[i]] = hidden[i].detach().cpu().numpy()
            if ids[i] in new_val_samples:
                correct_predictions.append(ids[i])
            else:
                if (label[i] == maxprob[1][i] and maxprob[0][i] > 0.5):
                    correct_predictions.append(ids[i])
                else:
                    incorrect_predictions.append(ids[i])

        # item.append(hidden) # the hidden layer will be the input to our new model

    # 2 BUILD A NEW MODEL TO PREDICT WHETHER VALIDATION ITEMS WERE CORRECT OR INCORRECT
    correct_model = NeuralNetwork_transfer(num_labels).to(device)
    correct_model.train()
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(correct_model.parameters(), lr=1e-4)

    # print(correct_predictions)

    dataset_2 = Dataset_transfer_learning(item_hidden_layers, correct_predictions, incorrect_predictions)
    dataloader_2 = DataLoader(dataset_2, batch_size=16, shuffle=True)
    for epoch in range(epochs):
        sumloss = 0

        # train the final layers model
        for batch_mini in dataloader_2:
            feature_vec = batch_mini[0].to(device)
            label = batch_mini[1].to(device)
            correct_model.zero_grad()

            logits, _ = correct_model(feature_vec)

            # compute loss function, do backward pass, and update the gradient
            loss = loss_function(logits, label)
            loss.backward()
            optimizer.step()
            sumloss += loss.item()
        # print(sumloss)
    correct_model.eval()
    trues = []
    predic = []
    for batch_mini in dataloader_2:
        feature_vec = batch_mini[0].to(device)
        label = batch_mini[1].to(device)
        logits, prob = correct_model(feature_vec)
        trues = trues + label.tolist()
        predic = predic + [(0 if x > 0.5 else 1) for x in prob[:, 0].tolist()]

    f1 = f1_score(trues, predic)
    print(f1)

    # 3 PREDICT WHETHER UNLABELED ITEMS ARE CORRECT

    deep_active_transfer_preds = []

    with torch.no_grad():
        dataset_unlabeled = Dataset_plants_free(unlabeled_data, path_to_dataset_train, model_feacher, device,
                                                path_to_dataset_numpy)
        unlabeled_dataloader = DataLoader(dataset_unlabeled, batch_size=32, shuffle=False)
        correct_model.eval()
        model.eval()

        for batch in unlabeled_dataloader:
            feature_vector = batch[0]

            # get prediction from main model
            # feature_vector = model_feacher.predict(imgs.to(device))
            hidden, logits2, probs = model(feature_vector)

            # use hidden layer from main model as input to model predicting correct/errors
            logits, probs = correct_model(hidden)

            # get confidence that item is correctly labeled
            # prob_correct = 1 - math.exp(log_probs.data.tolist()[0][1])

            # if (label == "0"):
            #     prob_correct = 1 - prob_correct

            # item[3] = "predicted_error"
            # item[4] = 1 - prob_correct
            for id, score in zip(batch[2], probs[:, 0]):
                s = [id, "predicted_error", score]
                deep_active_transfer_preds.append(s)

    deep_active_transfer_preds.sort(reverse=True, key=lambda x: x[2])

    return deep_active_transfer_preds[:select_per_epoch]

def get_atlas_samples( model, device, unlabeled_data, validation_data, feature_method, path_to_dir_train,
                        path_to_dataset_val, path_to_dataset_numpy, num_sample=200, num_labels=1):
    if (len(unlabeled_data) < num_sample):
        raise Exception('More samples requested than the number of unlabeled items')

    atlas_samples = []  # all items sampled by atlas
    new_val_samples = []  # all items sampled by atlas

    # print(number)
    while (len(atlas_samples) < num_sample):
        samples = get_deep_active_transfer_learning_uncertainty_samples(model, device, unlabeled_data, validation_data,
                                                                        feature_method, path_to_dir_train,
                                                                        path_to_dataset_val, path_to_dataset_numpy,
                                                                        new_val_samples,
                                                                        select_per_epoch=num_sample//4, epochs=130,
                                                                        num_labels=num_labels
                                                                        )
        for item in samples:
            atlas_samples.append([item[0], 'atlas', 0])
            unlabeled_data.remove(item[0])

            new_val_samples.append(item[0])  # append so that it is in the next iteration
        # model = train_model(feature_method, new_val_samples+labeled_data, validation_data, device)
        # model0, metricval_ao = train_model(feature_method, labeled_data, items_val, device, path_to_dataset_train,
        #                                    path_to_dataset_val, path_to_dataset_numpy, num_labels)

    return [x[0] for x in atlas_samples]

def for_api(rawmethods, device_arg, path_to_dataset_train, path_to_dataset_val, path_to_dataset_numpy,
            path_to_dataset_numpy_vae,
            train=True, rawnum_samples='100', check=False,
            num_train_for_model0=100, n_epoch=1, test_size=0.1, start_al=False, check_rnd=False, num_clusters=1):
    if device_arg == 'cuda':
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = 'cpu'

    model_feacher = Feature(device)
    model_feacher_vae = Feature_vae(device)
    dict_id, num_labels = read_dirs_dataset(path_to_dataset_train)

    all_items = [x for x in dict_id.keys()]

    random.seed(42)
    # del_labels = random.choices(list(range(num_labels)), k=100)
    del_labels = []
    labeled_data = sorted(prepare_items(limit=num_train_for_model0,
                                 train=True, seed=42, files=dict_id, test_size=test_size,
                                 del_labels=del_labels))

    labeled_data_first = copy.deepcopy(labeled_data)
    if test_size > 0:
        items_val = prepare_items(val=True, files=dict_id, test_size=test_size)
        unlabeled_data = list(set(all_items) - set(labeled_data) - set(items_val))
    else:
        dict_id_val, num_labels_val = read_dirs_dataset(path_to_dataset_val)
        items_val = prepare_items(limit=len(dict_id_val), train=True, seed=42, files=dict_id_val, test_size=0)
        unlabeled_data = list(set(all_items) - set(labeled_data))
    items_val = sorted(items_val)
    unlabeled_data = sorted(unlabeled_data)
    add_to_label_items = []

    use_first_model = False
    if use_first_model:
        if train:
            print('train zero model', end=' ')
            model0, f1 = train_model(model_feacher, labeled_data, items_val, device, path_to_dataset_train,
                                     path_to_dataset_val, path_to_dataset_numpy, num_labels)
            torch.save(model0.state_dict(), f'./models/model_weights_0.pth')
            print('f1 zero model {}'.format(f1))
        else:
            # print('load zero model', end=' ')
            model0 = NeuralNetwork(num_classes=num_labels).to(device)
            if num_train_for_model0 > 0:
                model0.load_state_dict(torch.load(f'./models/model_weights_0.pth'))


    methods = rawmethods.split('~')
    num_samples = [int(x.strip()) for x in rawnum_samples.split('~')]
    if start_al:
        for ne in range(n_epoch):
            # print('epoch {}, len label {}, len unleb {}'.format(ne+1, len(labeled_data), len(unlabeled_data)))
            unlabeled_data_copy = copy.deepcopy(unlabeled_data)
            for method, num_sample in zip(methods, num_samples):
                # AL
                if method == 'uncertainty_margin':
                    add_to_label_items = sampling_uncertainty(model0, device, model_feacher, unlabeled_data_copy,
                                                              method='margin',
                                                              num_sample=num_sample, path_to_dir=path_to_dataset_train,
                                                              num_labels=num_labels,
                                                              path_to_dataset_numpy=path_to_dataset_numpy)
                elif method == 'uncertainty_least':
                    add_to_label_items = sampling_uncertainty(model0, device, model_feacher, unlabeled_data_copy,
                                                              method='least',
                                                              num_sample=num_sample, path_to_dir=path_to_dataset_train,
                                                              num_labels=num_labels,
                                                              path_to_dataset_numpy=path_to_dataset_numpy)
                elif method == 'uncertainty_ratio':
                    add_to_label_items = sampling_uncertainty(model0, device, model_feacher, unlabeled_data_copy,
                                                              method='ratio',
                                                              num_sample=num_sample, path_to_dir=path_to_dataset_train,
                                                              num_labels=num_labels,
                                                              path_to_dataset_numpy=path_to_dataset_numpy)
                elif method == 'uncertainty_entropy':
                    add_to_label_items = sampling_uncertainty(model0, device, model_feacher, unlabeled_data_copy,
                                                              method='entropy',
                                                              num_sample=num_sample, path_to_dir=path_to_dataset_train,
                                                              num_labels=num_labels,
                                                              path_to_dataset_numpy=path_to_dataset_numpy)
                elif method == 'representative':
                    add_to_label_items = get_representative_samples(device, labeled_data, unlabeled_data, model_feacher,
                                                                    num_sample=num_sample, path_to_dir=path_to_dataset_train)
                elif method == 'cluster':
                    add_to_label_items = get_cluster_samples(device, unlabeled_data_copy, model_feacher,
                                                             num_clusters=num_clusters,
                                                             max_epochs=20,
                                                             num_sample=num_sample, path_to_dir=path_to_dataset_train,
                                                             path_to_dataset_numpy=path_to_dataset_numpy)
                elif method == 'new_cluster':
                    add_to_label_items = get_new_cluster_samples(device, unlabeled_data_copy, model_feacher_vae,
                                                             num_clusters=num_clusters,
                                                             num_sample=num_sample, path_to_dir=path_to_dataset_train,
                                                             path_to_dataset_numpy=path_to_dataset_numpy_vae)
                elif method == 'atlas':
                    add_to_label_items = get_atlas_samples(model0, device, unlabeled_data_copy, items_val,
                                                           model_feacher,
                                                           num_sample=num_sample,
                                                           path_to_dir_train=path_to_dataset_train,
                                                           path_to_dataset_val = path_to_dataset_val,
                                                           path_to_dataset_numpy=path_to_dataset_numpy,
                                                           num_labels=num_labels,
                                                           )
                elif method == 'representative_cluster':
                    add_to_label_items = get_representative_cluster_samples(device, labeled_data, unlabeled_data_copy,
                                                                            model_feacher,
                                                                            num_sample=num_sample,
                                                                            num_clusters=num_clusters,
                                                                            max_epochs=30,
                                                                            path_to_dir_train=path_to_dataset_train,
                                                                            path_to_dataset_val=path_to_dataset_val,
                                                                            path_to_dataset_numpy=path_to_dataset_numpy,
                                                                            )

                unlabeled_data_copy = add_to_label_items
            labeled_data = list(set(labeled_data + add_to_label_items))
            unlabeled_data = list(set(unlabeled_data) - set(add_to_label_items))

            # print('train model', end=' ')
            model0, metricval_ao = train_model(model_feacher, labeled_data, items_val, device, path_to_dataset_train,
                                               path_to_dataset_val, path_to_dataset_numpy, num_labels)
            torch.save(model0.state_dict(), f'./models/model_{ne+1}.pth')
            # print('f1 model {}'.format(f1))

    outdict = {'data': add_to_label_items}
    if check:
        if start_al:
            labeled_data2 = labeled_data
            # labeled_data2 = labeled_data + [x[0] for x in add_to_label_items]
            print('train al model. len label {}.'.format(len(labeled_data2)), end=' ')
            # _, metricval_ao = train_model(model_feacher, labeled_data2, items_val, device, path_to_dataset, num_labels)
            print('f1 al model {}'.format(metricval_ao))
            outdict['metricval_ao'] = metricval_ao

        # RND
        if check_rnd:
            random.seed(None)
            if sum(num_samples) == 0:
                labeled_data2 = labeled_data_first
            else:
                labeled_data2 = labeled_data_first + random.sample(unlabeled_data, k=n_epoch*sum(num_samples))
            # print('train rnd model. len label {}.'.format(len(labeled_data2)), end=' ')
            _, metricval_rnd = train_model(model_feacher, labeled_data2, items_val, device, path_to_dataset_train,
                                           path_to_dataset_val, path_to_dataset_numpy,
                                           num_labels)
            print('f1 rnd model {}'.format(metricval_rnd))

            outdict['metricval_rnd'] = metricval_rnd
    return outdict


if __name__ == '__main__':
    # path_to_dataset = '/home/neptun/PycharmProjects/datasets/plants/new plant diseases dataset(augmented)/New Plant ' \
    #                   'Diseases Dataset(Augmented)/train'
    # path_to_dataset = '/home/neptun/PycharmProjects/datasets/kkanji2'
    # path_to_dataset_train = '/media/alex/DAtA2/Datasets/flower_data/train'
    # path_to_dataset_val = '/media/alex/DAtA2/Datasets/flower_data/valid'
    # path_to_dataset_numpy = '/media/alex/DAtA2/Datasets/flower_data/numpy'
    # path_to_dataset_train = '/home/neptun/PycharmProjects/datasets/flower_data (копия)/train'
    # path_to_dataset_val = '/home/neptun/PycharmProjects/datasets/flower_data (копия)/valid'
    # path_to_dataset_numpy = '/home/neptun/PycharmProjects/datasets/flower_data (копия)/numpy'
    # path_to_dataset_numpy_vae = '/home/neptun/PycharmProjects/datasets/flower_data (копия)/numpy_vae'

    path_to_dataset_train = '/home/neptun/PycharmProjects/datasets/celeba/train'
    path_to_dataset_val = '/home/neptun/PycharmProjects/datasets/celeba/train'
    path_to_dataset_numpy = '/home/neptun/PycharmProjects/datasets/celeba/numpy'
    path_to_dataset_numpy_vae = ''


    k = 10
    # listnum = (['20000']*k + ['100']*k)
    listnum = (['200'] )
    # listnum = ['100']
    # list_methods = ['new_cluster',]
    # list_methods = ['uncertainty_margin',  'uncertainty_entropy', 'uncertainty_least', 'uncertainty_ratio',]
    num_f1 = []
    for i, num in enumerate(listnum):
    # for method in list_methods:
        # t_start = time.time()
        for j in range(k):
            # num_clusters = 40
            print(num, j, end=' ')
            outdict = for_api('', 'cuda', path_to_dataset_train, path_to_dataset_val, path_to_dataset_numpy,
                    path_to_dataset_numpy_vae,
                    train=False, num_train_for_model0=10,
                    rawnum_samples=num, test_size=0.2, n_epoch=1, check=True, start_al=False, check_rnd=True,
                    num_clusters=0)
            num_f1.append(outdict['metricval_rnd'])
            # t_stop = time.time()
            # print('time work sec: {}'.format(t_stop - t_start))
        # if (i + 1) % k == 0:
        print(num_f1)
        num_f1 = []
