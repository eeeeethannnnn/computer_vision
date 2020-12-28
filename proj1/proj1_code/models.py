"""
PyTorch tutorial on constructing neural networks:
https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def create_1D_Gaussian_kernel(standard_deviation: float) -> torch.FloatTensor:
    """Creates a 1D Gaussian kernel using the specified standard deviation.

    Note: ensure that the value of the kernel sums to 1.

    Args:
        standard_deviation (float): standard deviation of the gaussian

    Returns:
        torch.FloatTensor: required kernel as a row vector
    """
    kernel = torch.FloatTensor()
    #############################################################################
    # TODO: YOUR CODE HERE
    ############################################################################

    torch.set_printoptions(precision=8)
    torch.pi = torch.acos(torch.zeros(1)).item() *2
    size = int((4 * standard_deviation) + 1)
    kernel = torch.arange(size)
    mu = (size / 2) // 1
    denominator = float((standard_deviation ** 2) * 2)
    z = (torch.sqrt(torch.tensor(2 * torch.pi)) * standard_deviation)
    kernel = 1 / z * torch.exp(((kernel - mu) ** 2) / denominator * (-1))
    kernel = kernel / torch.sum(kernel)
    #raise NotImplementedError('`create_1D_Gaussian_kernel` function in ' + 'models.py needs to be implemented')
    #############################################################################
    #                             END OF YOUR CODE
    ############################################################################
    return kernel


def create_2D_Gaussian_kernel(standard_deviation: float) -> torch.FloatTensor:
    """Creates a 2D Gaussian kernel using the specified standard deviation in
    each dimension, and no cross-correlation between dimensions,

    i.e.
    sigma_matrix = [standard_deviation^2    0
                    0                       standard_deviation^2]


    The kernel should have:
    - shape (k, k) where k = standard_deviation * 4 + 1
    - mean = floor(k / 2)
    - values that sum to 1

    Args:
        standard_deviation (float): the standard deviation along a dimension

    Returns:
        torch.FloatTensor: 2D Gaussian kernel

    HINT:
    - The 2D Gaussian kernel here can be calculated as the outer product of two
      vectors drawn from 1D Gaussian distributions.
    """

    kernel_2d = torch.FloatTensor()


    #############################################################################
    # TODO: YOUR CODE HERE
    ############################################################################
    row = torch.Tensor()
    col = torch.Tensor()
    row = create_1D_Gaussian_kernel(standard_deviation)
    col = create_1D_Gaussian_kernel(standard_deviation)
    kernel_2d = torch.matmul(row.unsqueeze(-1), col.unsqueeze(-1).t())
    #raise NotImplementedError('`create_2D_Gaussian_kernel` function in ' + 'models.py needs to be implemented')
    #############################################################################
    #                             END OF YOUR CODE
    ############################################################################

    return kernel_2d


class HybridImageModel(nn.Module):
    def __init__(self):
        """
        Initializes an instance of the HybridImageModel class.
        """
        super(HybridImageModel, self).__init__()

    def get_kernel(self, cutoff_standarddeviation: int) -> torch.Tensor:
        """
        Returns a Gaussian kernel using the specified cutoff standard deviation.

        PyTorch requires the kernel to be of a particular shape in order to apply
        it to an image. Specifically, the kernel needs to be of shape (c, 1, k, k)
        where c is the # channels in the image. Start by getting a 2D Gaussian
        kernel using your implementation from Part 1, which will be of shape
        (k, k). Then, let's say you have an RGB image, you will need to turn this
        into a Tensor of shape (3, 1, k, k) by stacking the Gaussian kernel 3
        times.

        Args
        - cutoff_standarddeviation: int specifying the cutoff standard deviation
        Returns
        - kernel: Tensor of shape (c, 1, k, k) where c is # channels

        HINTS:
        - You will use the create_Gaussian_kernel() function from part1.py in this
          function.
        - Since the # channels may differ across each image in the dataset, make
          sure you don't hardcode the dimensions you reshape the kernel to. There
          is a variable defined in this class to give you channel information.
        - You can use torch.reshape() to change the dimensions of the tensor.
        - You can use torch's repeat() to repeat a tensor along specified axes.
        """
        kernel = torch.FloatTensor()

        ########################################################################
        #
        # TODO: YOUR CODE HERE
        ########################################################################
        k = create_2D_Gaussian_kernel(cutoff_standarddeviation)
        H, W = k.shape
        c = self.n_channels
        kernel = k.unsqueeze(0)
        kernel = kernel.unsqueeze(0)
        kernel = kernel.repeat(c, 1, 1, 1)
        ########################################################################
        #
        #                              END OF YOUR CODE
        ########################################################################

        return kernel

    def low_pass(self, x, kernel):
        """
        Applies low pass filter to the input image.

        Args:
        - x: Tensor of shape (b, c, m, n) where b is batch size
        - kernel: low pass filter to be applied to the image
        Returns:
        - filtered_image: Tensor of shape (b, c, m, n)

        HINT:
        - You should use the 2d convolution operator from torch.nn.functional.
        - Make sure to pad the image appropriately (it's a parameter to the
          convolution function you should use here!).
        - Pass self.n_channels as the value to the "groups" parameter of the
          convolution function. This represents the # of channels that the filter
          will be applied to.
        """

        filtered_image = torch.FloatTensor()

        ########################################################################
        #
        # TODO: YOUR CODE HERE
        ########################################################################

        pad = kernel.shape[2]
        half = (pad - 1) // 2
        filtered_image = F.conv2d(x, kernel, padding = (half, half), groups = self.n_channels)
        #raise NotImplementedError('`low_pass` function in `models.py` needs to ' + 'be implemented')
        ########################################################################
        #
        #                              END OF YOUR CODE
        ########################################################################

        return filtered_image

    def forward(self, image1, image2, cutoff_standarddeviation):
        """
        Takes two images and creates a hybrid image. Returns the low frequency
        content of image1, the high frequency content of image 2, and the hybrid
        image.

        Args
        - image1: Tensor of shape (b, c, m, n)
        - image2: Tensor of shape (b, c, m, n)
        - cutoff_standarddeviation: Tensor of shape (b)
        Returns:
        - low_frequencies: Tensor of shape (b, m, n, c)
        - high_frequencies: Tensor of shape (b, m, n, c)
        - hybrid_image: Tensor of shape (b, m, n, c)

        HINTS:
        - You will use the get_kernel() function and your low_pass() function in
          this function.
        - Similar to Part 1, you can get just the high frequency content of an
          image by removing its low frequency content.
        - Don't forget to make sure to clip the pixel values >=0 and <=1. You can
          use torch.clamp().
        - If you want to use images with different dimensions, you should resize
          them in the HybridImageDataset class using torchvision.transforms.
        """
        self.n_channels = image1.shape[1]

        low_frequencies = torch.FloatTensor()
        high_frequencies = torch.FloatTensor()
        hybrid_image = torch.FloatTensor()

        ########################################################################
        #
        # TODO: YOUR CODE HERE
        ########################################################################
        std = cutoff_standarddeviation.item()
        filter = self.get_kernel(std)
        low_frequencies = self.low_pass(image1, filter)
        high_frequencies = image2 - self.low_pass(image2, filter)
        tmp = (low_frequencies + high_frequencies)
        hybrid_image = torch.clamp(tmp.float(), min = 0, max = 1.0)
        ########################################################################
        #
        #                              END OF YOUR CODE
        ########################################################################

        return low_frequencies, high_frequencies, hybrid_image
