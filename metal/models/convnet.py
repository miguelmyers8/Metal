from metal.nn import NeuralNetwork

class ConvNet(NeuralNetwork):
    """docstring for CovNet."""

    def __init__(self, optimizer, loss, validation_data=None):
        super(ConvNet, self).__init__(optimizer, loss, validation_data=validation_data)
