import torch
import torch.nn as nn
from torchvision.models import alexnet


class MyAlexNet(nn.Module):
  def __init__(self):
    '''
    Init function to define the layers and loss function

    Note: Use 'sum' reduction in the loss_criterion. Read Pytorch documention to understand what it means

    Note: Do not forget to freeze the layers of alexnet except the last one. Otherwise the training will take a long time. To freeze a layer, set the
    weights and biases of a layer to not require gradients.

    Note: Map elements of alexnet to self.cnn_layers and self.fc_layers.

    Note: Remove the last linear layer in Alexnet and add your own layer to 
    perform 15 class classification.

    Note: Download pretrained alexnet using pytorch's API (Hint: see the import statements)
    '''
    super().__init__()

    self.cnn_layers = nn.Sequential()
    self.fc_layers = nn.Sequential()
    self.loss_criterion = None

    ############################################################################
    # Student code begin
    ############################################################################
    alex = alexnet(pretrained=True)

    self.cnn_layers.add_module('alexnet_feature', nn.Sequential(*(list(alex.children())[0])))
    """
      (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)) #64 @ 55*55
      (1): ReLU(inplace=True)
      (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False) # 64 @ 27*27
      (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)) # 192 @ 27* 27
      (4): ReLU(inplace=True)
      (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False) # 192 @ 14* 14
      (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) #384 @ 14*14
      (7): ReLU(inplace=True)
      (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) #256 @ 14*14
      (9): ReLU(inplace=True)
      (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) #256 @ 14*14
      (11): ReLU(inplace=True)
      (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    """
    for param in self.cnn_layers.parameters():
      param.requires_grad = False
    #self.cnn_layers.add_module('alexnet_avgpool', nn.AdaptiveAvgPool2d((6, 6)))
    #no flatten torch.Size([32, 256, 6, 6])
    #self.fc_layers.add_module('flatten', nn.Flatten())
    #self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
    self.fc_layers.add_module('alexnet_classifier', nn.Sequential(*(list(alex.children())[-1])))
    for param in self.fc_layers.parameters():
      param.requires_grad = False
    self.fc_layers.add_module('linear', nn.Linear(1000, 15))
    
    self.loss_criterion = nn.CrossEntropyLoss()
    #raise NotImplementedError('AlexNet not implemented')

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
    x = x.repeat(1, 3, 1, 1)  # as AlexNet accepts color images
    ############################################################################
    # Student code begin
    ############################################################################
    #raise NotImplementedError('forward function not implemented')
    conv_out = self.cnn_layers(x)
    #print(conv_out.shape)
    #avgpool_out = self.avgpool(conv_out)
    #print(avgpool_out.shape)
    flatten = torch.flatten(conv_out, 1)
    #flatten = avgpool_out.view(avgpool_out.size(0), 256 * 6 * 6)
    #print(flatten.shape)
    model_output = self.fc_layers(flatten)
    ############################################################################
    # Student code end
    ############################################################################

    return model_output
