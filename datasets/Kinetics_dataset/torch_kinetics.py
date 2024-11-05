from torch.utils.data import Dataset
from torchvision import datasets, transforms, models


root_folder = ''
dataset_train = datasets.Kinetics(split='train', num_classes=600, download=True)
