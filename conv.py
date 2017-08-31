import numpy as np
import random
import torch

class Conv2D:
    kernel_3s = torch.Tensor([
        [[-1, -1, -1], [0, 0, 0], [1, 1, 1]],
        [[-1,  0,  1], [-1, 0, 1], [-1, 0, 1]],
        [[ 1,  1,  1], [1, 1, 1], [1, 1, 1]]
    ])
    kernel_5s = torch.Tensor([
        [[-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
        [[-1, -1, 0, 1, 1], [-1, -1, 0, 1, 1], [-1, -1, 0, 1, 1], [-1, -1, 0, 1, 1], [-1, -1, 0, 1, 1]]
    ])
    def __init__(self, in_channel, o_channel, kernel_size, stride=1, mode="known"):
        self.in_channel = in_channel
        self.o_channel = o_channel
        self.kernel_to_use = None
        self.kernel_size = kernel_size
        self.stride = stride
        if mode == "rand":
            if kernel_size == 3:
                selected_idx = torch.LongTensor(random.sample(range(len(self.kernel_3s)), o_channel))
                self.kernel_to_use = torch.index_select(self.kernel_3s, 0, selected_idx)
            if kernel_size == 5:
                selected_idx = torch.LongTensor(random.sample(range(len(self.kernel_5s)), o_channel))
                self.kernel_to_use = torch.index_select(self.kernel_5s, 0, selected_idx)
        else:
            selected_idx = torch.LongTensor(range(o_channel))
            if kernel_size == 3:
                self.kernel_to_use = torch.index_select(self.kernel_3s, 0, selected_idx)
            if kernel_size == 5:
                self.kernel_to_use = torch.index_select(self.kernel_5s, 0, selected_idx)

    def forward(self, input_image):
        ops = 0
        c, h, w = input_image.size()
        h_output = h-self.kernel_size+1
        ops += 2
        w_output = w-self.kernel_size+1
        ops += 2
        reshape_size = np.power(self.kernel_size, 2)
        ops += 1
        output_image = torch.FloatTensor(self.o_channel, c, h_output, w_output).zero_()
        for kid, kernel in enumerate(self.kernel_to_use):
            kernel_1d = kernel.contiguous().view(reshape_size)
            for i in range(c):
                for j in range(0, h_output, self.stride):
                    for k in range(0, w_output, self.stride):
                        block = input_image[i][j:j+self.kernel_size, k:k+self.kernel_size].contiguous().view(reshape_size)
                        ops += 2
                        output_image[kid][i][j][k] += torch.dot(kernel_1d, block)
                        ops += 2 * kernel_1d.size()[0]
            # torch.div(output_image[kid], c, out=output_image[kid])
        return (ops, output_image)
