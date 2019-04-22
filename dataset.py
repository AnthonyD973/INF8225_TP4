import sys
import torch
import torchvision
import random
import noise

class RandomResizeDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, ratio_mu = 0.0, ratio_sigma = 1.0, seed=None):
        self._dataset = dataset
        self._ratio_mu = ratio_mu
        self._ratio_sigma = ratio_sigma
        if seed is None:
            seed = random.random()
        self._seed = seed
        self._rng = noise.snoise2

    def __getitem__(self, index):
        data, _ = self._dataset[index]
        rng_state = random.getstate()
        random.seed(int(((self._rng(self._seed, float(index))+1)/2) * sys.maxsize))
        ratio = pow(2, -abs(random.gauss(self._ratio_mu, self._ratio_sigma)))
        random.setstate(rng_state)
        transformations = torchvision.transforms.Compose([
            torchvision.transforms.Resize((int(ratio * data.size[0]), int(ratio * data.size[1]))),
            torchvision.transforms.Resize(data.size),
            torchvision.transforms.ToTensor()
        ])
        transformed = transformations(data).to(torch.double)
        label = torchvision.transforms.ToTensor()(data).to(torch.double)
        if transformed.size(0) == 1:
            transformed = transformed.repeat(3, 1, 1)
            label = label.repeat(3, 1, 1)
        return transformed, label

    def __len__(self):
        return len(self._dataset)