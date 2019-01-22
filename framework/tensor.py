
# Tensors are an abstract form of Vectors and Matrices

#  each tensor gets attributes.
#  creators is a list which contains any tensors used in the creation of the current tensor (which
#  defaults to None).

#  when we add the two tensors x and y together, z has two creators, x
#  and y. creation_op is a related feature which stores the instructions the creators used in the
#  creation process.

#   when we perform z = x + y, we have actualy created what is called
#   a computation graph which has three nodes (x, y and z) and 2 edges (z->x and z->y). Each
#   edge is labeled by the creation_op "add". This graph allows us to recursively backpropgate
#   gradients



import numpy as np


class Tensor (object):

    def __init__(self, data, creators = None, creation_op = None):
        self.data = np.array(data)
        self.creation_op = creation_op
        self.creators = creators
        self.grad = None

    def __add__(self, other):
        return  Tensor(self.data + other.data,
                        creators = [self, other],
                        creation_op = "add")

    def __repr__(self):
        return str(self.data.__repr__())

    def __str__(self):
        return str(self.data.__str__())


    def backward(self, grad):
        self.grad = grad

        if(self.creation_op == "add"):
            self.creators[0].backward(grad)
            self.creators[1].backward(grad)
