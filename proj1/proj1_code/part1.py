import torch

def my_1dfilter(signal: torch.FloatTensor,
                kernel: torch.FloatTensor) -> torch.FloatTensor:
    """Filters the signal by the kernel.

    output = signal * kernel where * denotes the cross-correlation function.
    Cross correlation is similar to the convolution operation with difference
    being that in cross-correlation we do not flip the sign of the kernel.

    Reference:
    - https://mathworld.wolfram.com/Cross-Correlation.html
    - https://mathworld.wolfram.com/Convolution.html

    Note:
    1. The shape of the output should be the same as signal.
    2. You may use zero padding as required. Please do not use any other
       padding scheme for this function.
    3. Take special care that your function performs the cross-correlation
       operation as defined even on inputs which are asymmetric.

    Args:
        signal (torch.FloatTensor): input signal. Shape=(N,)
        kernel (torch.FloatTensor): kernel to filter with. Shape=(K,)

    Returns:
        torch.FloatTensor: filtered signal. Shape=(N,)
    """
    filtered_signal = torch.FloatTensor()

    #############################################################################
    # TODO: YOUR CODE HERE
    ############################################################################
    signal_len = len(signal)
    kernel_len = len(kernel)
    #print(signal_len)
    #print(kernel_len)
    half = (kernel_len) // 2
    signal_pad = torch.nn.functional.pad(signal, (half, kernel_len - half))
    #print(signal_pad.shape)
    filtered_signal = torch.empty(signal_len)
    #print(filtered_signal.shape)
    for i in range(0, signal_len):
        #print(i)
        res = torch.dot(signal_pad[i : i + kernel_len], kernel)
        filtered_signal[i] = res
    #print(filtered_signal)
    #raise NotImplementedError
    #############################################################################
    #                             END OF YOUR CODE
    ############################################################################

    return filtered_signal
