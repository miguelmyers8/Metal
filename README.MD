# Metal V2
Pytorch clone with a Auto differentiation framework for deep learning

## Requirements
python 3.6

## Using Tensors
```python
    from metal.tensor import Tensor as mt

    a = mt([1,2,3,4,5], requires_grad=True)
    b = mt([2,2,2,2,2], requires_grad=True)
    c = mt([5,4,3,2,1], requires_grad=True)
    d = a + b
    e = b + c
    f = d + e
    f.backward()
    assert(b.grad.data.tolist() == np.array([2,2,2,2,2]).tolist())
    b.zero_grad()
```
