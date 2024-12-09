import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as F_v2
import math

def softclamp_upper(x: torch.Tensor, upper: float):
    return upper - F.softplus(upper - x)

def softclamp(x: torch.Tensor, upper: float, lower: float):
    return lower + F.softplus(x - lower) - F.softplus(x - upper)

def cosine_schedule(base, end, T):
    return end - (end - base) * ((torch.arange(0, T, 1) * math.pi / T).cos() + 1) / 2
    
def sample_action(p=0.25, dtype=torch.float32, device=torch.device('cpu')):
    action = torch.rand(5, dtype=dtype, device=device) * 2 - 1
    mask = torch.rand(action.shape, dtype=torch.float32, device=device) < p
    return action * mask

def transform(images, action):
    angle = action[0].item() * 180
    translate_x = action[1].item() * 8
    translate_y = action[2].item() * 8
    scale = action[3].item() * 0.25 + 1.0
    shear = action[4].item() * 25
    return F_v2.affine(images, angle=angle, translate=(translate_x, translate_y), scale=scale, shear=shear)

def interact(images, groups:int=8):
    if groups > images.shape[0]:
        groups = images.shape[0]
    transformed = []
    actions = []
    N, group_size = images.shape[0], images.shape[0] // groups
    for lo in range(0, N, group_size):
        hi = min(lo + group_size, N)
        action = sample_action(dtype=images.dtype, device=images.device)
        transformed.append(transform(images[lo:hi], action))
        actions.append(action.repeat(hi-lo, 1))
    return torch.cat(transformed, dim=0), torch.cat(actions, dim=0)

def embed_dataset(model, dataset):
    device = next(model.parameters()).device
    batch_size = 256
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embeddings = torch.empty((len(dataset), model.z_dim), device=device)
    labels = torch.empty(len(dataset), dtype=torch.long, device=device)

    for i, (x, y) in enumerate(dataloader):
        x = x.to(device)
        emb = model.embed(x)
        lo, hi = i*len(x), min((i+1)*len(x), len(dataset))
        embeddings[lo:hi] = emb.detach()
        labels[lo:hi] = y

    return embeddings, labels

def get_balanced_subset(data, labels, n_train: int, shuffle=True):
    n_per_class = n_train // 10
    train_data, train_labels, test_data, test_labels = [], [], [], []
    for i in range(10):
        indices = torch.where(labels == i)[0]
        if shuffle:
            shuffle_idx = torch.randperm(len(indices))
            indices = indices[shuffle_idx]
        train_data.append(data[indices[:n_per_class]])
        train_labels.append(labels[indices[:n_per_class]])
        test_data.append(data[indices[n_per_class:]])
        test_labels.append(labels[indices[n_per_class:]])

    train_data = torch.cat(train_data)
    train_labels = torch.cat(train_labels)
    test_data = torch.cat(test_data)
    test_labels = torch.cat(test_labels)

    return train_data, train_labels, test_data, test_labels