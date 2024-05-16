import os, sys
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from cnn import Net   
from attn import MHA


def mnist_vis(data_loader):
    
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(data_loader.dataset.data[i], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(data_loader.dataset.targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()


def main(config):

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                    ])

    dataset1 = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=config['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset1, batch_size=config['batch_size'], shuffle=True)

    # model = Net()
    model = MHA(64, 8, 10)

    # test MHA
    # test_model = MHA(64, 8, 10)
    # sample = torch.randn(16, 1, 28, 28)
    # output = test_model(sample)
    # print(output.shape)
    # end test MHA

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(config['num_epochs']):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            outputs, scores = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, config['num_epochs'], i+1, len(train_loader), loss.item()))
        
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                outputs, scores = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

        torch.save(model.state_dict(), 'models/MHA.ckpt')

if __name__ == '__main__':
    config = {}
    exec(open('cfg.py').read(), config)
    main(config)
    
