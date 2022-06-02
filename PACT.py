import warnings
import torch
import numpy as np
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_





class PACT_ReLU(torch.autograd.Function):
    """
    def __init__(self, ALPHA):
        super(PACT_ReLU, self).__init__()
        self.alpha = ALPHA
    """

    @staticmethod
    def forward(ctx, input, alpha):
        y = 0.5*(torch.abs(input)-torch.abs(input-alpha)+alpha)
        output = torch.round(y * (2**8 - 1) / alpha) * (alpha / (2**8 - 1))
        ctx.save_for_backward(input, alpha)
        #return torch.clamp(input, 0, alpha.data)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        print(ctx.saved_tensors)
        input, alpha, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_alpha = grad_output.clone()
        #print('2. ', grad_input)
        print('3. ', grad_output.shape, grad_output)
        grad_input[grad_input.le(0)] = 0
        grad_input[grad_input.ge(alpha)] = 0
        #grad_output = torch.zeros_like(grad_output)
        print('4. ', grad_output.shape, grad_output)
        #print('5. ', alpha)

        #if grad_input < alpha:
        #    grad_output = 0

        return grad_input, torch.sum(grad_alpha), None


class PACT(torch.nn.Module):
    def __init__(self, _Alpha = 10.0):
        super(PACT, self).__init__()
        self._Alpha = torch.nn.Parameter(torch.Tensor([_Alpha]))
        #print('1. ',self.Alpha.shape)

    def forward(self, input):

        return PACT_ReLU.apply(input, self._Alpha)
"""
class PACT_ReLU(torch.autograd.Function):

    def __init__(self, alpha):
        super(PACT_ReLU, self).__init__()
        self.alpha = alpha
        #alpha = Parameter(torch.Tensor(1))

    def forward(self, input):
        y = 0.5*(np.absolute(input)-np.absolute(input-self.alpha)+self.alpha)
        output = np.round(y * (2**8 - 1) / self.alpha) * (self.alpha / (2**8 - 1))
        return output

    def backward(self, grad_output):
        grad_input = grad_output.clone()
        grad_output = 1
        if grad_input < self.alpha :
            grad_output = 0

        return grad_output


#if __name__ == '__main__':
"""