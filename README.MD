# Metal V2
Auto differentiation framework for deep learning

## Requirements
python 3.6

## Using Tensors
```python
    from autograd.tensor import Tensor as at

    a = at([1,2,3,4,5], requires_grad=True)
    b = at([2,2,2,2,2], requires_grad=True)
    c = at([5,4,3,2,1], requires_grad=True)
    d = a + b
    e = b + c
    f = d + e
    f.backward(Tensor(np.array([1,1,1,1,1])))
    assert(b.grad.data.tolist() == np.array([2,2,2,2,2]).tolist())
    b.zero_grad()
```
