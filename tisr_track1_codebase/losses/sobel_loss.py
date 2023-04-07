import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import LOSS_REGISTRY


class GradLayer(nn.Module):

    def __init__(self):
        super(GradLayer, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def get_gray(self,x):
        ''' 
        Convert image to its gray one.
        '''
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = x.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        x_gray = x.mul(convert).sum(dim=1)
        return x_gray.unsqueeze(1)

    def forward(self, x):

        if x.shape[1] == 3:
            x = self.get_gray(x)

        x_v = F.conv2d(x, self.weight_v, padding=1)
        x_h = F.conv2d(x, self.weight_h, padding=1)
        x = torch.sqrt(torch.pow(x_v, 2) + torch.pow(x_h, 2) + 1e-6)

        return x


@LOSS_REGISTRY.register()
class SobelLoss(nn.Module):

    def __init__(self, loss_weight=1.0):
        super(SobelLoss, self).__init__()
        self.loss = nn.L1Loss()
        self.grad_layer = GradLayer()
        self.loss_weight = loss_weight

    def forward(self, output, gt_img):
        output_grad = self.grad_layer(output)
        gt_grad = self.grad_layer(gt_img)
        return self.loss(output_grad, gt_grad) * self.loss_weight


if __name__ == "__main__":

    import cv2
    import numpy as np
    
    net = GradLayer()
    # a = img.shape # (256, 256, 3)    
    # img = (img / 255.0).astype(np.float32)
    img = np.random.rand(256, 256, 3)
    print(img.shape)
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
    img = net(img) # input img: data range [0, 1]; data type torch.float32; data shape [1, 3, 256, 256]
    b = img.shape # torch.Size([1, 1, 256, 256])
    img = (img[0, :, :, :].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    print(img.shape)
