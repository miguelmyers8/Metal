import numpy as _np
from metal.utils.functions import accuracy_score,  is_binary, is_stochastic
from abc import ABC, abstractmethod
from metal.autograd import numpy as np


class ObjectiveBase(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def loss(self, y_true, y_pred):
        pass

    @abstractmethod
    def acc(self,  y_true, y_pred):
        pass


class CrossEntropy(ObjectiveBase):
    def __init__(self):
        """
        A cross-entropy loss.
        Notes
        -----
        For a one-hot target **y** and predicted class probabilities
        :math:`\hat{\mathbf{y}}`, the cross entropy is
        .. math::
                \mathcal{L}(\mathbf{y}, \hat{\mathbf{y}})
                    = \sum_i y_i \log \hat{y}_i
        """
        super().__init__()

    def __call__(self, y, y_pred):
        return self.loss(y, y_pred)

    def __str__(self):
        return "CrossEntropy"

    @staticmethod
    def loss(y, y_pred):
        """
        Compute the cross-entropy (log) loss.
        Notes
        -----
        This method returns the sum (not the average!) of the losses for each
        sample.
        Parameters
        ----------
        y : :py:class:`ndarray <numpy.ndarray>` of shape (n, m)
            Class labels (one-hot with `m` possible classes) for each of `n`
            examples.
        y_pred : :py:class:`ndarray <numpy.ndarray>` of shape (n, m)
            Probabilities of each of `m` classes for the `n` examples in the
            batch.
        Returns
        -------
        loss : float
            The sum of the cross-entropy across classes and examples.
        """
        #is_binary(y)
        #is_stochastic(y_pred)

        # prevent taking the log of 0
        eps = np.finfo(float).eps
        N = y_pred.shape[0]

        # each example is associated with a single class; sum the negative log
        # probability of the correct label over all samples in the batch.
        # observe that we are taking advantage of the fact that y is one-hot
        # encoded
        cross_entropy = -np.sum(y * np.log(y_pred + eps))
        return cross_entropy / np.float32(N)

    @staticmethod
    def acc(y, p):
        return accuracy_score(_np.argmax(y, axis=1), _np.argmax(p, axis=1))
