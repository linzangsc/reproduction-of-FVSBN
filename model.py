import torch
import torch.nn as nn

class LogisticRegression(nn.Module):
    def __init__(self, image_size, device):
        super(LogisticRegression, self).__init__()
        self.device = device
        self.image_size = image_size
        self.linear = nn.Linear(image_size**2, image_size**2)
        for row, weights in enumerate(self.linear.weight.data):
            weights[row:].data.fill_(0)

    def forward(self, image):
        # image: (batch_size, 1, height, width)
        batch_size, _, image_height, image_width = image.shape
        image = image.reshape(batch_size, -1)
        theta = self.linear(image)
        return nn.functional.sigmoid(theta)

    def zero_grad_for_extra_weights(self):
        for row, grads in enumerate(self.linear.weight.grad):
            grads[row: ] = 0
    
    def sample(self, sample_num):
        with torch.no_grad():
            sampled_image = torch.zeros((sample_num, self.image_size**2), dtype=torch.float32).to(self.device)
            sampled_theta = torch.zeros((sample_num, self.image_size**2), dtype=torch.float32).to(self.device)
            for pixel_index in range(self.image_size**2):
                weight = self.linear.weight.data[pixel_index]
                bias = self.linear.bias.data[pixel_index]
                cur_theta = nn.functional.sigmoid(torch.matmul(sampled_image, weight) + bias)
                sampled_image[:, pixel_index] = torch.bernoulli(cur_theta)
                sampled_theta[:, pixel_index] = cur_theta
        return sampled_image, sampled_theta
