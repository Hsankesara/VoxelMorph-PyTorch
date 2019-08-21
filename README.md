# VoxelMorph-PyTorch

An unofficial PyTorch implementation of VoxelMorph- An unsupervised 3D deformable image registration method.

## Image registration

Image registration is the process of aligning two images. For that purpose, one image is taken as a fixed image and the other one is moving image. The goal is to apply a transformation to moving image such that the transformed image(known as the registered image) has the same orientation as the fixed image. The application of the process is vast. The major application of this problem is in medical imaging where two different types of images(like MRI and CT scan) of the same object need to be aligned properly for better understanding.

There are two types of algorithms in image registration. First is Rigid Image Registration(RIR) and the second is Deformation Image Registration (DIR). The process in which all transformations are affine that is the pixel to pixel relationship remains the same as before is known as RIR. This is a linear method and frequently used in the past. It is useful when the moving image has no deformity. The major drawback of this method is that it cannot be used when the moving image incurred some deformation. This happens quite often in medical images when there is a disease like a tumor which can grow or shrink with time. Deformation image registration(DIR) process is used in such cases.

DIR methods are employed when RIR cannot perform the desired task. They can be used to analysis and comparison of medical structures between the scans. Such analysis is used to assess and understand the evolution of brain anatomy over time for individuals with the disease. Deformable registration strategies often involve two steps: an initial affine transformation for global alignment, followed by a much slower deformable transformation with more degrees of freedom. We concentrate on the latter step, in which we compute a dense, nonlinear correspondence for all pixels.

Since the problem is highly ill-posed and has vast applications hence it became a perfect problem for deep learning algorithms to solve. Many different architectures has been proposed but recently [VoxelMorph](https://arxiv.org/abs/1809.05231) has been proposed which surpassed the prior state of the art. Since, VoxelMorph only has Tensorflow implementation hence I've developed an unoficial PyTorch implementation along with an easy to use API.

## How to use

```python
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

    ## Main code
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

```

## Resources

1. [Know more about image registration](https://www.sciencedirect.com/topics/neuroscience/image-registration)
2. [Approaches to Registering Images](https://www.mathworks.com/help/images/approaches-to-registering-images.html)
3. [QuickSilver: A fast deformable image registration technique](https://arxiv.org/pdf/1703.10908.pdf)
4. [VoxelMorph](https://arxiv.org/abs/1809.05231)
5. [Spatial Transformer Networks](https://arxiv.org/pdf/1506.02025.pdf)

## Author

**[Heet Sankesara](https://github.com/Hsankesara)**

[<img src="http://i.imgur.com/0o48UoR.png" width="35" padding="10" margin="10">](https://github.com/Hsankesara/) [<img src="https://i.imgur.com/0IdggSZ.png" width="35" padding="10" margin="10">](https://www.linkedin.com/in/heet-sankesara-72383a152/) [<img src="http://i.imgur.com/tXSoThF.png" width="35" padding="10" margin="10">](https://twitter.com/heetsankesara3) [<img src="https://loading.io/s/icon/vzeour.svg" width="35" padding="10" margin="10">](https://www.kaggle.com/hsankesara)
