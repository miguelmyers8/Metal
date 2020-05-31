# Autodiff
This is a reimplementation/reengineering based on [the full version of Autograd](https://github.com/hips/autograd).<br>
for me details click the installation link.

## installation
python 3.6 + <br>
['autograd'](https://github.com/miguelmyers8/autodiff)

## The goal:
To dive into Matthew Johnson autograd package, understand it the best I can, document, and reimplement.<br>
This autograd will function like pytorch.

## Example:
```python
from autograd.numpy.container import Container
import numpy as _np
import autograd.numpy as anp

x = Container(_np.linspace(-7,7,2),False)
i = Container(_np.linspace(-3,3,2),True)

p = 9+i*x+2/x
anp.sum(p).backward()
i.grad
```

## Extending Container_
the Container_ class is meant to function as the baseclass for any of your subclass if you wish to extend the opertions in Metal. <br>
