from ..core import DropoutFunction
from ..module.module import Module

class Dropout(Module):
    """docstring for Dropout."""

    def __init__(self,ratio=0.5,**kwargs):
        super(Dropout, self).__init__()
        self.ratio = ratio


    def forward(self,x):
        mask = None
        return_mask = False

        if self.training:
            func = DropoutFunction(self.ratio,mask,return_mask)
            out = func.forward_cpu(x)
            mask = func.mask
        else:
            out = x
            mask = None

        if return_mask:
            return out, mask
        return out

Dropout.register()
