import numpy as _np
from metal.autograd import numpy as np



def calc_fan(weight_shape):
    """
    Compute the fan-in and fan-out for a weight matrix/volume.
    Parameters
    ----------
    weight_shape : tuple
        The dimensions of the weight matrix/volume. The final 2 entries must be
        `in_ch`, `out_ch`.
    Returns
    -------
    fan_in : int
        The number of input units in the weight tensor
    fan_out : int
        The number of output units in the weight tensor
    """
    if len(weight_shape) == 2:
        fan_in, fan_out = weight_shape
    elif len(weight_shape) in [3, 4]:
        in_ch, out_ch = weight_shape[2:]
        kernel_size = _np.prod(weight_shape[:2])
        fan_in, fan_out = in_ch * kernel_size, out_ch * kernel_size
    else:
        raise ValueError("Unrecognized weight dimension: {}".format(weight_shape))
    return fan_in, fan_out



def he_uniform(weight_shape):
    """
    Initializes network weights `W` with using the He uniform initialization
    strategy.
    Notes
    -----
    The He uniform initializations trategy initializes thew eights in `W` using
    draws from Uniform(-b, b) where
    .. math::
        b = \sqrt{\\frac{6}{\\text{fan_in}}}
    Developed for deep networks with ReLU nonlinearities.
    Parameters
    ----------
    weight_shape : tuple
        The dimensions of the weight matrix/volume.
    Returns
    -------
    W : :py:class:`ndarray <numpy.ndarray>` of shape `weight_shape`
        The initialized weights.
    """
    fan_in, fan_out = calc_fan(weight_shape)
    b = np.sqrt(6 / fan_in)
    return _np.random.uniform(-b, b, size=weight_shape)


def he_normal(weight_shape):
    """
    Initialize network weights `W` using the He normal initialization strategy.
    Notes
    -----
    The He normal initialization strategy initializes the weights in `W` using
    draws from TruncatedNormal(0, b) where the variance `b` is
    .. math::
        b = \\frac{2}{\\text{fan_in}}
    He normal initialization was originally developed for deep networks with
    :class:`~numpy_ml.neural_nets.activations.ReLU` nonlinearities.
    Parameters
    ----------
    weight_shape : tuple
        The dimensions of the weight matrix/volume.
    Returns
    -------
    W : :py:class:`ndarray <numpy.ndarray>` of shape `weight_shape`
        The initialized weights.
    """
    fan_in, fan_out = calc_fan(weight_shape)
    std = np.sqrt(2 / fan_in)
    return truncated_normal(0, std, weight_shape)


def glorot_uniform(weight_shape, gain=1.0):
    """
    Initialize network weights `W` using the Glorot uniform initialization
    strategy.
    Notes
    -----
    The Glorot uniform initialization strategy initializes weights using draws
    from ``Uniform(-b, b)`` where:
    .. math::
        b = \\text{gain} \sqrt{\\frac{6}{\\text{fan_in} + \\text{fan_out}}}
    The motivation for Glorot uniform initialization is to choose weights to
    ensure that the variance of the layer outputs are approximately equal to
    the variance of its inputs.
    This initialization strategy was primarily developed for deep networks with
    tanh and logistic sigmoid nonlinearities.
    Parameters
    ----------
    weight_shape : tuple
        The dimensions of the weight matrix/volume.
    Returns
    -------
    W : :py:class:`ndarray <numpy.ndarray>` of shape `weight_shape`
        The initialized weights.
    """
    fan_in, fan_out = calc_fan(weight_shape)
    b = gain * _np.sqrt(6 / (fan_in + fan_out))
    return _np.random.uniform(-b, b, size=weight_shape)


def glorot_normal(weight_shape, gain=1.0):
    """
    Initialize network weights `W` using the Glorot normal initialization strategy.
    Notes
    -----
    The Glorot normal initializaiton initializes weights with draws from
    TruncatedNormal(0, b) where the variance `b` is
    .. math::
        b = \\frac{2 \\text{gain}^2}{\\text{fan_in} + \\text{fan_out}}
    The motivation for Glorot normal initialization is to choose weights to
    ensure that the variance of the layer outputs are approximately equal to
    the variance of its inputs.
    This initialization strategy was primarily developed for deep networks with
    :class:`~numpy_ml.neural_nets.activations.Tanh` and
    :class:`~numpy_ml.neural_nets.activations.Sigmoid` nonlinearities.
    Parameters
    ----------
    weight_shape : tuple
        The dimensions of the weight matrix/volume.
    Returns
    -------
    W : :py:class:`ndarray <numpy.ndarray>` of shape `weight_shape`
        The initialized weights.
    """
    fan_in, fan_out = calc_fan(weight_shape)
    std = gain * np.sqrt(2 / (fan_in + fan_out))
    return truncated_normal(0, std, weight_shape)


def truncated_normal(mean, std, out_shape):
    """
    Generate draws from a truncated normal distribution via rejection sampling.
    Notes
    -----
    The rejection sampling regimen draws samples from a normal distribution
    with mean `mean` and standard deviation `std`, and resamples any values
    more than two standard deviations from `mean`.
    Parameters
    ----------
    mean : float or array_like of floats
        The mean/center of the distribution
    std : float or array_like of floats
        Standard deviation (spread or "width") of the distribution.
    out_shape : int or tuple of ints
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.
    Returns
    -------
    samples : :py:class:`ndarray <numpy.ndarray>` of shape `out_shape`
        Samples from the truncated normal distribution parameterized by `mean`
        and `std`.
    """
    samples = _np.random.normal(loc=mean, scale=std, size=out_shape)
    reject = np.logical_or(samples >= mean + 2 * std, samples <= mean - 2 * std)
    while any(reject.flatten()):
        resamples = _np.random.normal(loc=mean, scale=std, size=reject.sum())
        samples[reject] = resamples
        reject = np.logical_or(samples >= mean + 2 * std, samples <= mean - 2 * std)
    return samples
