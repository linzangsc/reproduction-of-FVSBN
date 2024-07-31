import os
import yaml
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import CustomizedDataset, visualize_binary_result, sigmoid, visualize_float_result
from model import LogisticRegression
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, config) -> None:
        self.config = config
        self.image_size = config['image_size']
        self.logger = SummaryWriter(self.config['log_path'])
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = config['batch_size']
        self.num_epochs = config['num_epochs']
        self.dataset = CustomizedDataset()
        self.train_dataset = self.dataset.train_dataset
        self.test_dataset = self.dataset.test_dataset
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, 
                                                        batch_size=self.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, 
                                                       batch_size=self.batch_size, shuffle=False)
        self.model = LogisticRegression(image_size=28, device=self.device).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])

    def log_likelihood_calc(self, images, theta, eps=1e-8):
        # theta shape: (batch_size, 784), images shape: (batch_size, 1, 28, 28)
        images = images.view(images.shape[0], -1)
        log_likelihood = images*torch.log(theta+eps) + (1-images)*torch.log(1-theta+eps)
        return -log_likelihood.sum() / images.shape[0]

    def train(self):
        self.model.train()
        for epoch in range(self.num_epochs):
            for i, (images, labels) in enumerate(self.train_loader):
                # translate to binary images
                images = torch.round(images)
                images = images.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images).to(self.device)
                loss = self.log_likelihood_calc(images, outputs)
                loss.backward()
                self.model.zero_grad_for_extra_weights()
                self.optimizer.step()
                if i % 100 == 0:
                    print(f'Epoch [{epoch+1}/{self.num_epochs}], Step [{i+1}/{len(self.train_loader)}], Loss: {loss.item():.4f}')
                self.logger.add_scalar('loss/train', loss.item(), i + epoch * len(self.train_loader))
            if (epoch) % 10 == 0:
                sampled_image, sampled_theta = self.model.sample(16)
                self.visualize(sampled_image, sampled_theta)
                self.save_model(self.config['ckpt_path'])

    def save_model(self, output_path):
        if not os.path.exists(output_path): os.mkdir(output_path)
        torch.save(self.model.state_dict(), os.path.join(output_path, f"model.pth"))

    def visualize(self, sampled_image, sampled_theta):
        sampled_image = sampled_image.reshape(sampled_image.shape[0], self.image_size, self.image_size).to('cpu')
        npy_sampled_image = np.array(sampled_image, dtype=np.int32)
        sampled_theta = sampled_theta.reshape(sampled_theta.shape[0], self.image_size, self.image_size).to('cpu')
        npy_sampled_theta = np.array(sampled_theta)
        visualize_binary_result(npy_sampled_image, self.config['visualization_path'])
        visualize_float_result(npy_sampled_theta, self.config['visualization_path'])

if __name__ == "__main__":
    with open('./config.yaml') as f:
        config = yaml.safe_load(f)
    trainer = Trainer(config=config)
    trainer.train()
