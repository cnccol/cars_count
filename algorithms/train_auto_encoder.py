import random

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from auto_encoder import AutoEncoder as AE

IMAGE_SIZE = 784
IMAGE_WIDTH = IMAGE_HEIGHT = 28
CODE_SIZE = 20
NUM_EPOCHS = 5
BATCH_SIZE = 128
LR = 0.002
optimizer_cls = optim.Adam

# Load data
train_data = datasets.MNIST('~/data/mnist/', train=True , transform=transforms.ToTensor())
test_data  = datasets.MNIST('~/data/mnist/', train=False, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE, num_workers=4, drop_last=True)

# Instantiate model
autoencoder = AE(CODE_SIZE)
loss_fn = nn.BCELoss()
optimizer = optimizer_cls(autoencoder.parameters(), lr=LR)

# Training loop
for epoch in range(NUM_EPOCHS):
    print("Epoch %d" % epoch)
    
    for i, (images, _) in enumerate(train_loader):    # Ignore image labels
        out, code = autoencoder(Variable(images))
        
        optimizer.zero_grad()
        loss = loss_fn(out, images)
        loss.backward()
        optimizer.step()
        
    print("Loss = %.3f" % loss.data[0])

# Try reconstructing on test data
test_image = random.choice(test_data)
test_image = Variable(test_image.view([1, 1, IMAGE_WIDTH, IMAGE_HEIGHT]))
test_reconst, _ = autoencoder(test_image)

torchvision.utils.save_image(test_image.data, 'orig.png')
torchvision.utils.save_image(test_reconst.data, 'reconst.png')