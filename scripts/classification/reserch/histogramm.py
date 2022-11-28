import yaml
from autoencoder.PyTorch_VAE.models import *
from autoencoder.PyTorch_VAE.experiment import VAEXperiment
from pytorch_lightning.utilities.seed import seed_everything
from autoencoder.PyTorch_VAE.dataset import VAEDataset
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

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

    def loss_function(self, *args, **kwargs):
        return self.model.loss_function(*args, **kwargs)

if __name__ == '__main__':
    with open(path_yaml_celeba, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    config['exp_params']['M_N'] = config['exp_params']['kld_weight']
    seed_everything(config['exp_params']['manual_seed'], True)

    config['data_params']['train_batch_size'] = 1
    vae = Feature_vae('cuda:0')
    data_celeba = VAEDataset(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0,
                      limit=5_000, filter_label=0)
    data_celeba.setup()

    err = []
    train_dataset = data_celeba.train_dataloader()
    for x, l in train_dataset:
        args = vae.predict(x.to('cuda:0'))
        loss = vae.loss_function(*args, **config['exp_params'])
        err.append(loss['loss'].item())


    data_celeba4 = VAEDataset(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0,
                      limit='5_000:50_000', filter_label=0)
    data_celeba4.setup()

    err4 = []
    train_dataset4 = data_celeba4.train_dataloader()
    for x, l in train_dataset4:
        args = vae.predict(x.to('cuda:0'))
        loss = vae.loss_function(*args, **config['exp_params'])
        err4.append(loss['loss'].item())


    data_celeba2 = VAEDataset(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0,
                      limit=50_000, filter_label=1)
    data_celeba2.setup()

    err2 = []
    train_dataset2 = data_celeba2.train_dataloader()

    for x, l in train_dataset2:
        args = vae.predict(x.to('cuda:0'))
        loss = vae.loss_function(*args, **config['exp_params'])
        err2.append(loss['loss'].item())

    plt.imshow(np.array(transforms.ToPILImage()(args[0][0])))
    plt.show()
    plt.imshow(np.array(transforms.ToPILImage()(args[1][0])))
    plt.show()


    with open(path_yaml_plants, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    config['exp_params']['M_N'] = config['exp_params']['kld_weight']
    config['data_params']['train_batch_size'] = 1
    data_plants = VAEDataset(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0,
                      limit=-1, filter_label=None)
    data_plants.setup()

    err3 = []
    train_dataset = data_plants.train_dataloader()
    for x, l in train_dataset:
        args = vae.predict(x.to('cuda:0'))
        loss = vae.loss_function(*args, **config['exp_params'])
        err3.append(loss['loss'].item())
    #
    # plt.imshow(np.array(transforms.ToPILImage()(args[0][0])))
    # plt.show()
    # plt.imshow(np.array(transforms.ToPILImage()(args[1][0])))
    # plt.show()


    plt.hist(err, bins=50, alpha=0.3)
    plt.hist(err2, bins=50, alpha=0.3)
    plt.hist(err3, bins=50, alpha=0.3)
    plt.hist(err4, bins=50, alpha=0.3)
    plt.show()