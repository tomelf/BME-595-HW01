from conv import Conv2D
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage, Compose, Normalize
# from matplotlib import pyplot as plt
import torch

def main():
    input_image = Image.open("test.png")
    transform = Compose([
        ToTensor(),
    ])
    input_tensor = transform(input_image.convert('RGB'))
    # print input_tensor

    parameters = [
        {"in_channel": 3, "o_channel": 1, "kernel_size": 3, "stride":1, "mode": 'known'},
        {"in_channel": 3, "o_channel": 2, "kernel_size": 5, "stride":1, "mode": 'known'},
        {"in_channel": 3, "o_channel": 3, "kernel_size": 3, "stride":1, "mode": 'known'},
    ]
    for parameter in parameters:
        conv2d = Conv2D(in_channel=parameter["in_channel"],
                    o_channel=parameter["o_channel"],
                    kernel_size=parameter["kernel_size"],
                    stride=parameter["stride"],
                    mode=parameter["mode"])
        Number_of_ops, out_tensor = conv2d.forward(input_tensor)
        # print out_tensor
        # print Number_of_ops
        for idx, t in enumerate(out_tensor):
            transform = Compose([
                ToPILImage(),
            ])
            output_image = transform(t)

            # t_np = t.numpy()
            # print t_np
            # output_image = Image.fromarray(t_np, 'RGB')

            # plt.imshow(output_image)
            # plt.show()

            output_image.save('test_{0:d}_{1:d}_{2:d}_{3:d}_{4}_{5:d}.png'.format(
                    parameter["in_channel"],
                    parameter["o_channel"],
                    parameter["kernel_size"],
                    parameter["stride"],
                    parameter["mode"],
                    idx+1
                )
            )
            # Number_of_ops, output_image = conv2d.forward(input_image)
            # Assert output_image.size() == expected_image.size()

if __name__ == "__main__":
    main()
