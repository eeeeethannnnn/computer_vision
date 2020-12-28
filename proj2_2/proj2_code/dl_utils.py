'''
Utilities to be used along with the deep model
'''

import torch


def predict_labels(model_output: torch.tensor) -> torch.tensor:
    '''
    Predicts the labels from the output of the model.

    Args:
    -   model_output: the model output [Dim: (N, 15)]
    Returns:
    -   predicted_labels: the output labels [Dim: (N,)]
    '''

    predicted_labels = None
    ############################################################################
    # Student code begin
    ############################################################################
    predicted_labels = model_output.argmax(dim = 1)
    # raise NotImplementedError('predict_labels not implemented')
    ############################################################################
    # Student code end
    ############################################################################

    return predicted_labels


def compute_loss(model: torch.nn.Module,
                 model_output: torch.tensor,
                 target_labels: torch.tensor,
                 is_normalize: bool = True) -> torch.tensor:
    '''
    Computes the loss between the model output and the target labels

    Note: we have initialized the loss_criterion in the model with the sum
    reduction.

    Args:
    -   model: model (which inherits from nn.Module), and contains loss_criterion
    -   model_output: the raw scores output by the net [Dim: (N, 15)]
    -   target_labels: the ground truth class labels [Dim: (N, )]
    -   is_normalize: bool flag indicating that loss should be divided by the
                      batch size
    Returns:
    -   the loss value
    '''
    loss = None

    ############################################################################
    # Student code begin
    ############################################################################
    loss_function = model.loss_criterion
    #print(len(model_output))
    #print(len(target_labels))
    
    loss = loss_function(model_output, target_labels)
    if is_normalize:
        loss = loss / len(model_output)
    #print(type(loss))
    # raise NotImplementedError('compute_loss not implemented')
    ############################################################################
    # Student code end
    ############################################################################

    return loss
