from conv import Conv2D
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage, Compose, Normalize
import matplotlib.pyplot as plt
import torch
import time
from scipy import misc
import numpy as np

def main():
    input_filename = "test-1920x1080"
    input_image = Image.open("{0}.png".format(input_filename))
    transform = Compose([
        ToTensor(),
    ])
    input_tensor = transform(input_image.convert('RGB'))
    print input_tensor.size()

    run_parts = [True, False, False]

    # PartA
    if run_parts[0]:
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
            print out_tensor.size()
            for idx, t in enumerate(out_tensor):
                misc.imsave('{6}_{0:d}_{1:d}_{2:d}_{3:d}_{4}_{5:d}.png'.format(
                        parameter["in_channel"],
                        parameter["o_channel"],
                        parameter["kernel_size"],
                        parameter["stride"],
                        parameter["mode"],
                        idx+1,
                        input_filename,
                    ),
                    t.numpy()
                )

    # PartB
    if run_parts[1]:
        parameters = []
        running_times = []
        o_channels = [np.power(2, i) for i in range(11)]
        for o_channel in o_channels:
            parameters.append({"in_channel": 3, "o_channel": o_channel, "kernel_size": 3, "stride":1, "mode": 'rand'})
        for parameter in parameters:
            conv2d = Conv2D(in_channel=parameter["in_channel"],
                        o_channel=parameter["o_channel"],
                        kernel_size=parameter["kernel_size"],
                        stride=parameter["stride"],
                        mode=parameter["mode"])
            start_time = time.time()
            Number_of_ops, out_tensor = conv2d.forward(input_tensor)
            running_times.append(time.time() - start_time)
        radius = range(1, len(running_times)+1)
        print radius
        print running_times
        plt.ylabel('Conv2D.forward() running time')
        plt.plot(radius, running_times)
        plt.show()

    # PartC
    if run_parts[2]:
        running_ops = []
        parameters = []
        kernel_sizes = [3, 5, 7, 9, 11]
        for k in kernel_sizes:
            parameters.append({"in_channel": 3, "o_channel": 2, "kernel_size": k, "stride":1, "mode": 'rand'})
        for parameter in parameters:
            conv2d = Conv2D(in_channel=parameter["in_channel"],
                        o_channel=parameter["o_channel"],
                        kernel_size=parameter["kernel_size"],
                        stride=parameter["stride"],
                        mode=parameter["mode"])
            Number_of_ops, out_tensor = conv2d.forward(input_tensor)
            running_ops.append(Number_of_ops)
        radius = kernel_sizes
        print radius
        print running_ops
        fig, ax = plt.subplots()
        plt.ylabel('Conv2D.forward() running operations')
        plt.plot(radius, running_ops, '.')
        plt.show()

if __name__ == "__main__":
    main()
