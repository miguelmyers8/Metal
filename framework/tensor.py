
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

    def __init__(self, data, autograd = False, creators = None, creation_op = None, id = None):

        self.data = np.array(data)
        self.creation_op = creation_op
        self.creators = creators
        self.grad = None
        self.autograd = autograd
        self.children = {}
        if(id is None):
            id = np.random.randint(0,100000)
        self.id = id

        #   Keep track of how many children
        if(creators is not None):
            for c in creators:
                if(self.id not in c.children):
                    c.children[self.id] = 1
                else:
                    c.children[self.id] += 1

    #  Check whether a Tensor has received the correct #
    #  of gradients from each child.
    def all_children_grads_accounted_for(self):
        for id,cnt in self.children.items():
            if(cnt != 0):
                return False
        return True

    # begin actual backpropagation
    def __add__(self, other):
        if(self.autograd and other.autograd):
            return  Tensor(self.data + other.data,
                            autograd = True,
                            creators = [self, other],
                            creation_op = "add")
        return Tensor(self.data + other.data)

    def __repr__(self):
        return str(self.data.__repr__())


    def __str__(self):
        return str(self.data.__str__())


    def backward(self, grad = None, grad_origin = None):
        if(self.autograd):
             # Check to make sure we can backpropagate or
             # whether we are still waiting for a gradient,
             # in which case decrement
             # counter
            if(grad_origin is not None):
                if(self.children[grad_origin.id] == 0):
                    raise Exception("cannot backpropgate more than once")
                else:
                    self.children[grad_origin.id] -= 1
            if(self.grad is None):
                # accumulate gradients from several children
                self.grad = grad
            else:
                self.grad += grad

            if(self.creators is not None and (self.all_children_grads_accounted_for() or grad_origin is None)):
                if(self.creation_op == "add"):
                    self.creators[0].backward(self.grad, self)
                    self.creators[1].backward(self.grad, self)
