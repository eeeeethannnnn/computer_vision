import torch
import torch.nn as nn


class SimpleNet(nn.Module):
    '''Simple Network with atleast 2 conv2d layers and two linear layers.'''

    def __init__(self):
        '''
        Init function to define the layers and loss function

        Note: Use 'sum' reduction in the loss_criterion. Read Pytorch documention to understand what it means

        Hints:
        1. Refer to https://pytorch.org/docs/stable/nn.html for layers
        2. Remember to use non-linearities in your network. Network without
           non-linearities is not deep.
        3. You will get 3D tensor for an image input from self.cnn_layers. You need
           to process it and make it a compatible tensor input for self.fc_layers.
        '''
        super().__init__()

        self.cnn_layers = nn.Sequential()  # conv2d and supporting layers here
        self.fc_layers = nn.Sequential()  # linear and supporting layers here
        self.loss_criterion = None

        ############################################################################
        # Student code begin
        ############################################################################
        #input 1 @ 64x64
        #conv: 5x5 kernel
        # feature 10 @ 60*60
        self.cnn_layers.add_module('conv1', nn.Conv2d(1, 10, (5, 5)))
        #pool 3*3 kernel
        self.cnn_layers.add_module('pool1', nn.MaxPool2d(3))
        self.cnn_layers.add_module('relu1', nn.ReLU())
        #feature 10 @ 20*20

        #input 10 @ 20*20
        #conv: 5*5 kernel
        #feature 20 @ 16*16
        self.cnn_layers.add_module('conv2', nn.Conv2d(10, 20, (5, 5)))
        #pool 3*3 kernel
        self.cnn_layers.add_module('pool2', nn.MaxPool2d(3))
        self.cnn_layers.add_module('relu2', nn.ReLU())
        #feature 20 @ 5*5

        #flatten
        self.cnn_layers.add_module('flatten', nn.Flatten())
        #dim 500 fully connected
        #self.fc_layers.add_module('linear1', nn.Linear(20 * 5 * 5, 500))
        #self.fc_layers.add_module('relu1', nn.ReLU())
        #dim 100 fully connected
        self.fc_layers.add_module('linear1', nn.Linear(500, 100))
        self.fc_layers.add_module('relu1', nn.ReLU())
        # dim 15 fully connected
        self.fc_layers.add_module('linear2', nn.Linear(100, 15))
        # raise NotImplementedError('SimpleNet not initialized')
        self.loss_criterion = nn.CrossEntropyLoss()
        ############################################################################
        # Student code end
        ############################################################################

    def forward(self, x: torch.tensor) -> torch.tensor:
        '''
        Perform the forward pass with the net

        Note: do not perform soft-max or convert to probabilities in this function

        Args:
        -   x: the input image [Dim: (N,C,H,W)]
        Returns:
        -   y: the output (raw scores) of the net [Dim: (N,15)]
        '''
        model_output = None
        ############################################################################
        # Student code begin
        ############################################################################
        conv_out = self.cnn_layers(x)
        model_output = self.fc_layers(conv_out)
        # raise NotImplementedError('forward function not implemented')
        ############################################################################
        # Student code end
        ############################################################################

        return model_output
