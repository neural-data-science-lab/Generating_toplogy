import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.dataloader as dataloader

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import os
import time
import matplotlib.pyplot as plt
from PIL import Image

num_epochs = 20
batch_size = 100# train dataset

n_dim = 3
latent_dim = 2

# generate original pointsw
num_orig_samples = 10
class PointDataset(Dataset):

    def __init__(self, num_points, n):
        self.num_points = num_points

        self.data = torch.rand(num_points, n)
        print('self.data.size()', self.data.size())

    def __len__(self):
        return len(self.num_points)

    def __getitem__(self, idx):
        return self.data[idx, :]

data_loader = dataloader.DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=True,
)

data_iter = iter(data_loader)  # data_loader is iterable

class AutoEncoder(nn.Module):
    def __init__(self, in_dim=3, hidden_size=2, out_dim=3):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=hidden_size),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=out_dim),
            nn.Sigmoid()
        )

    def forward(self, *input):
        out = self.encoder(*input)
        out = self.decoder(out)
        return out

autoEncoder = AutoEncoder(in_dim=in_dim, hidden_size=latent_dim, out_dim=in_dim)
if torch.cuda.is_available():
    autoEncoder.cuda() 

Loss = nn.BCELoss()
Optimizer = optim.Adam(autoEncoder.parameters(), lr=0.001)

for epoch in range(num_epochs):
    t_epoch_start = time.time()
    for i, (image_batch, _) in enumerate(data_loader):
        # flatten batch
        if torch.cuda.is_available():
            image_batch = image_batch.cuda()
        predict = autoEncoder(image_batch)

        Optimizer.zero_grad()
        loss = Loss(predict, image_batch)
        loss.backward()
        Optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch {}/{}, Iter {}/{}, loss: {:.4f}, time: {:.2f}s'.format(
                epoch + 1, num_epochs, (i + 1), len(dataset) // batch_size, loss.data, time.time() - t_epoch_start
            ))

    # # show image
    # img_reconstructed = Image.open(filename)
    # plt.figure()
    # plt.subplot(1,2,1),plt.title('real_images')
    # plt.imshow(image_real), plt.axis('off')
    # plt.subplot(1,2,2), plt.title('reconstructed_images')
    # plt.imshow(img_reconstructed), plt.axis('off')
    # plt.show()
