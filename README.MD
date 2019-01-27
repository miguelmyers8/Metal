# Metal V2
Auto differentiation framework for deep learning

## Requirements
python 3.6

## Using Tensors
```python
    a = Tensor([1,2,3,4,5], autograd=True)
    b = Tensor([2,2,2,2,2], autograd=True)
    c = Tensor([5,4,3,2,1], autograd=True)
    d = a + b
    e = b + c
    f = d + e
    f.backward(Tensor(np.array([1,1,1,1,1])))
    assert(b.grad.data.tolist(),np.array([2,2,2,2,2]).tolist())
```
