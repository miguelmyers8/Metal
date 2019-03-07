import numpy as np
from metal.tensor import Tensor

class Parameter(Tensor):
    def __init__(self, input_size=None, output_size=None, inputs=None) -> None:

        if input_size and output_size is not None:
            np.random.seed(input_size)
            data = np.random.randn(input_size, output_size) * np.sqrt(2/output_size)

        elif inputs is not None:
            data = inputs

        super().__init__(data, requires_grad=True)
