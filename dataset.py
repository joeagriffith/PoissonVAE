import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm

def remove_to_tensor(transform):
    if type(transform) == transforms.ToTensor:
        transform = None

    if type(transform) == transforms.Compose:
        new_transforms = []
        for t in transform.transforms:
            if type(t) != transforms.ToTensor:
                new_transforms.append(t)
        transform = transforms.Compose(new_transforms)
    return transform

class MNIST(Dataset):
    def __init__(self, root, train=True, device=torch.device('cpu'), normalize=True, dtype=torch.float32, fashion=False):

        if root is not None:
            transform = [
                transforms.ToTensor(),
            ]
            if normalize:
                transform.append(transforms.Normalize((0.1307,), (0.3081,)))
            transform = transforms.Compose(transform)
            if fashion:
                dataset = datasets.FashionMNIST(root=root, train=train, transform=transform, download=False)
            else:
                dataset = datasets.MNIST(root=root, train=train, transform=transform, download=False)
            data = []
            targets = []
            loop = tqdm(range(len(dataset)), leave=False)
            for i in loop:
                d, t = dataset[i]
                if type(t) is not torch.Tensor:
                    t = torch.tensor(t)
                data.append(d)
                targets.append(t)
                
            assert type(data[0]) == torch.Tensor, print(f"Data is {type(data[0])} not torch.Tensor")
            assert type(targets[0]) == torch.Tensor, print(f"Targets is {type(targets[0])} not torch.Tensor")
            
            self.shape = data[0].shape
            self.device = device
            self.images = torch.stack(data).to(device, dtype=dtype)
            self.targets = torch.stack(targets).to(device, dtype=torch.long)
            
    #  Now a man who needs no introduction
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]        
    
    # Randomly shuffles the dataset and returns a validation set, removing the validation set from the original dataset
    def get_val_dataset(self, val_ratio: float, shuffle: bool):
        assert 0 < val_ratio < 1, "Validation ratio must be between 0 and 1"

        if shuffle:
            perm = torch.randperm(len(self))
            self.images = self.images[perm]
            self.targets = self.targets[perm]

        n_val = int(len(self) * val_ratio)
        n_train = len(self) - n_val

        val_dataset = MNIST(root=None)
        val_dataset.shape = self.shape
        val_dataset.device = self.device
        val_dataset.images = self.images[n_train:]
        val_dataset.targets = self.targets[n_train:]

        self.images = self.images[:n_train]
        self.targets = self.targets[:n_train]
        return val_dataset