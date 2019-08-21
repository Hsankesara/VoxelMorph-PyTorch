import voxelmorph2d as vm2d
import voxelmorph3d as vm3d
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import os
from skimage.transform import resize
import multiprocessing as mp
from tqdm import tqdm
import gc
import time
from sklearn.model_selection import train_test_split
from matplotlib.lines import Line2D
use_gpu = torch.cuda.is_available()


class VoxelMorph():
    """
    VoxelMorph Class is a higher level interface for both 2D and 3D
    Voxelmorph classes. It makes training easier and is scalable.
    """

    def __init__(self, input_dims, is_2d=False, use_gpu=False):
        self.dims = input_dims
        if is_2d:
            self.vm = vm2d
            self.voxelmorph = vm2d.VoxelMorph2d(input_dims[0] * 2, use_gpu)
        else:
            self.vm = vm3d
            self.voxelmorph = vm3d.VoxelMorph3d(input_dims[0] * 2, use_gpu)
        self.optimizer = optim.SGD(
            self.voxelmorph.parameters(), lr=1e-4, momentum=0.99)
        self.params = {'batch_size': 3,
                       'shuffle': True,
                       'num_workers': 6,
                       'worker_init_fn': np.random.seed(42)
                       }
        self.device = torch.device("cuda:0" if use_gpu else "cpu")

    def check_dims(self, x):
        try:
            if x.shape[1:] == self.dims:
                return
            else:
                raise TypeError
        except TypeError as e:
            print("Invalid Dimension Error. The supposed dimension is ",
                  self.dims, "But the dimension of the input is ", x.shape[1:])

    def forward(self, x):
        self.check_dims(x)
        return voxelmorph(x)

    def calculate_loss(self, y, ytrue, n=9, lamda=0.01, is_training=True):
        loss = self.vm.vox_morph_loss(y, ytrue, n, lamda)
        return loss

    def train_model(self, batch_moving, batch_fixed, n=9, lamda=0.01, return_metric_score=True):
        self.optimizer.zero_grad()
        batch_fixed, batch_moving = batch_fixed.to(
            self.device), batch_moving.to(self.device)
        registered_image = self.voxelmorph(batch_moving, batch_fixed)
        train_loss = self.calculate_loss(
            registered_image, batch_fixed, n, lamda)
        train_loss.backward()
        self.optimizer.step()
        if return_metric_score:
            train_dice_score = self.vm.dice_score(
                registered_image, batch_fixed)
            return train_loss, train_dice_score
        return train_loss

    def get_test_loss(self, batch_moving, batch_fixed, n=9, lamda=0.01):
        with torch.set_grad_enabled(False):
            registered_image = self.voxelmorph(batch_moving, batch_fixed)
            val_loss = self.vm.vox_morph_loss(
                registered_image, batch_fixed, n, lamda)
            val_dice_score = self.vm.dice_score(registered_image, batch_fixed)
            return val_loss, val_dice_score


class Dataset(data.Dataset):
    """
    Dataset class for converting the data into batches.
    The data.Dataset class is a pyTorch class which help
    in speeding up  this process with effective parallelization
    """
    'Characterizes a dataset for PyTorch'

    def __init__(self, list_IDs):
        'Initialization'
        self.list_IDs = list_IDs

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        fixed_image = torch.Tensor(
            resize(io.imread('./fire-fundus-image-registration-dataset/' + ID + '_1.jpg'), (256, 256, 3)))
        moving_image = torch.Tensor(
            resize(io.imread('./fire-fundus-image-registration-dataset/' + ID + '_2.jpg'), (256, 256, 3)))
        return fixed_image, moving_image


def main():
    '''
    In this I'll take example of FIRE: Fundus Image Registration Dataset
    to demostrate the working of the API.
    '''
    vm = VoxelMorph(
        (3, 256, 256), is_2d=True)  # Object of the higher level class
    DATA_PATH = './fire-fundus-image-registration-dataset/'
    params = {'batch_size': 1,
              'shuffle': True,
              'num_workers': 6,
              'worker_init_fn': np.random.seed(42)
              }

    max_epochs = 2
    filename = list(set([x.split('_')[0]
                         for x in os.listdir('./fire-fundus-image-registration-dataset/')]))
    partition = {}
    partition['train'], partition['validation'] = train_test_split(
        filename, test_size=0.33, random_state=42)

    # Generators
    training_set = Dataset(partition['train'])
    training_generator = data.DataLoader(training_set, **params)

    validation_set = Dataset(partition['validation'])
    validation_generator = data.DataLoader(validation_set, **params)

    # Loop over epochs
    for epoch in range(max_epochs):
        start_time = time.time()
        train_loss = 0
        train_dice_score = 0
        val_loss = 0
        val_dice_score = 0
        for batch_fixed, batch_moving in training_generator:
            loss, dice = vm.train_model(batch_moving, batch_fixed)
            train_dice_score += dice.data
            train_loss += loss.data
        print('[', "{0:.2f}".format((time.time() - start_time) / 60), 'mins]', 'After', epoch + 1, 'epochs, the Average training loss is ', train_loss *
              params['batch_size'] / len(training_set), 'and average DICE score is', train_dice_score.data * params['batch_size'] / len(training_set))
        # Testing time
        start_time = time.time()
        for batch_fixed, batch_moving in validation_generator:
            # Transfer to GPU
            loss, dice = vm.get_test_loss(batch_moving, batch_fixed)
            val_dice_score += dice.data
            val_loss += loss.data
        print('[', "{0:.2f}".format((time.time() - start_time) / 60), 'mins]', 'After', epoch + 1, 'epochs, the Average validations loss is ', val_loss *
              params['batch_size'] / len(validation_set), 'and average DICE score is', val_dice_score.data * params['batch_size'] / len(validation_set))


if __name__ == "__main__":
    main()
