import torch

# from torchvision.models import efficientnet_b0
import random
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms as T
from engine import train_one_epoch, evaluate

from unit import Dataset_objdetect, prepare_items_od
import copy
import utils
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.faster_rcnn import GeneralizedRCNNTransform


# https://github.com/rmunro/pytorch_active_learning

with open('setting.yaml') as f:
    templates = yaml.safe_load(f)

random.seed(templates['randomseed'])
device = "cuda:0" if torch.cuda.is_available() else "cpu"

def get_transform():
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)

def train_model(items_train, items_val, label, num_epochs=5):

    ds0 = Dataset_objdetect(items_train, label=label, train=True, transforms=get_transform())
    ds_val = Dataset_objdetect(items_val, label=label, val=True, transforms=get_transform())

    train_dataloader = DataLoader(ds0, batch_size=32, shuffle=True, collate_fn=utils.collate_fn)
    val_dataloader = DataLoader(ds_val, batch_size=32, shuffle=True, collate_fn=utils.collate_fn)
    num_classes = 2

    model = get_model_instance_segmentation(num_classes)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    # params = [p for p in model.roi_heads.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=1e-4)
    best_score = 0
    best_model = None
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_freq=100)
        # if epoch == num_epochs - 1:
        metrics = evaluate(model, val_dataloader, device=device)
        if best_score < metrics.coco_eval['bbox'].stats[0]:
            best_score = metrics.coco_eval['bbox'].stats[0]
            best_model = copy.deepcopy(model)
            print('Best mAP = {:.03}'.format(best_score))

    print('end mAP {}'.format(best_score))

    return best_model

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.transform = GeneralizedRCNNTransform(min_size=224, max_size=224, image_mean=[0.485, 0.456, 0.406],
                                               image_std=[0.229, 0.224, 0.225])

    return model

def sampling_uncertainty(model, unlabeled_data, lable=None):
    with torch.no_grad():
        model.eval()
        dataset_train = Dataset_objdetect(unlabeled_data, label=lable, train=True, transforms=get_transform())
        train_dataloader = DataLoader(dataset_train, batch_size=16, shuffle=False, collate_fn=utils.collate_fn)
        indexs = []
        values = []
        for images, targets, indx in train_dataloader:
            images = list(img.to(device) for img in images)
            outputs = model(images)
            prob = [x['scores'].tolist() for x in outputs]
            confidence = []
            for b_row in prob:
                if len(b_row) == 0:
                    confidence.append(0)
                else:
                    dd = []
                    for s in b_row:
                        a = s
                        # if s > 0.5:
                        #     a = s
                        # else:
                        #     a = 1 - s
                        k = (1 - a) 
                        dd.append(k)
                    # confidence.append(sum(dd)/len(dd))
                    confidence.append(max(dd))

            indexs += [x for x in indx]
            values += confidence
            # print('{} of {}'.format(i, len(train_dataloader)))

    out_name = []
    alfa = 0.7
    temp = sorted(values)[-int(templates['num_for_al'] * alfa):]
    # out_value = []
    for ell in temp:
        indx = values.index(ell)
        # out_value.append(ell)
        out_name.append(indexs[indx])

    temp = sorted(values)[:-int(templates['num_for_al'] * alfa)]
    temp = random.sample(temp, k=int(templates['num_for_al'] * (1 - alfa)))
    # out_value = []
    for ell in temp:
        indx = values.index(ell)
        # out_value.append(ell)
        out_name.append(indexs[indx])
    return out_name


if __name__ == '__main__':
    current_label = 16
    # Counter({13: 531315, 12: 101709, 9: 60027, 15: 20118, 10: 12794, 6: 7960, 16: 5622, 3: 5544, 17: 2251, 7: 1506,
    #          11: 1446, 2: 525, 1: 511, 5: 412, 4: 330, 14: 263, 8: 21, 0: 3})
    # id_all = np.load(os.path.join('/home/neptun/PycharmProjects/dataset/dota/id.npy')).tolist()
    all_items = prepare_items_od(train=True, seed=templates['randomseed'])
    labeled_data = prepare_items_od(limit=templates['first'], train=True, seed=templates['randomseed'], lable=current_label)
    items_val = prepare_items_od(val=True, lable=current_label, seed=templates['randomseed'])

    model0 = train_model(labeled_data, items_val, label=current_label, num_epochs=templates['n_epoch'])
    torch.save(model0.state_dict(), f'models/model_weights_od_0.pth')

    # model0 = get_model_instance_segmentation(2).to(device)
    # model0.load_state_dict(torch.load(f'./models/model_weights_od_0.pth', map_location=device))

    unlabeled_data = list(set(all_items) - set(labeled_data))

    add_to_label_items = sampling_uncertainty(model0, unlabeled_data, lable=current_label)
    # print(add_to_label_items)
    # # add2_to_label_items = get_cluster_samples(add_to_label_items, num_clusters=10, max_epochs=10, num=200)
    # #
    print('AL')
    labeled_data2 = labeled_data + add_to_label_items
    train_model(labeled_data2, items_val, label=current_label, num_epochs=templates['n_epoch'])
    # #
    print('RND')
    labeled_data2 = labeled_data + random.sample(unlabeled_data, k=templates['num_for_al'])
    train_model(labeled_data2, items_val, label=current_label, num_epochs=templates['n_epoch'])

