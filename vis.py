import os, sys
import time
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton

from attn import MHA

class MHAVisualizer:
    def __init__(self) -> None:
        self.attn_map = None
        self.right_click = False
        self.left_click = False

    def on_move(self, event):
        if event.inaxes:
            print(f'data coords {event.xdata} {event.ydata}',
                f'axis coords {event.x} {event.y}')

    def on_click(self, event):
        if event.button is MouseButton.LEFT:
            print(f'data coords {event.xdata} {event.ydata}',
                  f'axis coords {event.x} {event.y}')
            self.x = int(event.xdata)
            self.y = int(event.ydata) 
            self.left_click = True
            
        if event.button is MouseButton.RIGHT:
            print('Move to next image')
            self.right_click = True

    def mha_vis(self, image, target, scores):
        self.scores = scores

        while True:
            plt.subplot(3,3,1)
            plt.imshow(image, cmap='gray', interpolation='none')
            plt.title("Ground Truth: {}".format(target))
            plt.xticks([])
            plt.yticks([])
            plt.connect('button_press_event', self.on_click)
            
            if self.right_click:
                self.right_click = False
                plt.close()
                break

            if self.left_click:
                plt.subplot(3, 3, 1)
                color_image = np.zeros((28, 28, 3))
                color_image[:, :, 0] = image
                color_image[:, :, 1] = image
                color_image[:, :, 2] = image
                color_image[self.y:self.y + 2, self.x:self.x + 2, 0] = 1
                color_image[self.y:self.y + 2, self.x:self.x + 2, 1] = 0
                color_image[self.y:self.y + 2, self.x:self.x + 2, 2] = 0
                plt.imshow(color_image, interpolation='none')

                for i in range(0, 8):
                    plt.subplot(3, 3, i+2)
                    attn_img = scores[0][i][int(self.y / 4) * 7 + int(self.x / 4)].detach().numpy()
                    attn_img = attn_img.reshape(7, 7)
                    plt.imshow(attn_img, cmap='gray', interpolation='none')
                    plt.title(f'Attention Map {i}')
                    self.left_click = False
            
            time.sleep(0.1)
            plt.show()
        

    def run(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                        ])

        dataset1 = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        dataset2 = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

        train_loader = torch.utils.data.DataLoader(dataset1, batch_size=1, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset1, batch_size=1, shuffle=True)

        model = MHA(64, 8, 10)
        model.load_state_dict(torch.load('models/MHA.ckpt'))

        for i, (images, labels) in enumerate(train_loader):
            print('Image ', i)
            outputs, scores = model(images)
            self.mha_vis(images[0][0], labels[0], scores)


if __name__ == '__main__':
    vis = MHAVisualizer()
    vis.run()