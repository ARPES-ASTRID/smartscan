import numpy as np


def ndim_aqfunc(x: np.ndarray, gp) -> float:
    """Compute the acquisition function for a given GP and a given point"""
    a = 2.5  #####change here, 3.0 for 95 percent confidence interval
    norm = 2.0
    ret = None
    for i in range(gp.input_dim - 1):
        x_pred = np.c_[x, np.zeros(x.shape[0]) + i].reshape(-1, gp.input_dim)
        cov = gp.posterior_covariance(x_pred)["v(x)"]
        if ret is None:
            ret = cov
        else:
            ret += cov

    ret = a * np.sqrt(ret)

    for i in range(gp.input_dim - 1):
        x_pred = np.c_[x, np.zeros(x.shape[0]) + i].reshape(-1, gp.input_dim)
        mean = gp.posterior_mean(x_pred)["f(x)"]
        ret += norm * mean

    return ret


def acquisition_function_nd(
    x: np.ndarray, gp, a: float, weights: np.array, norm: float = 2.0, c: float = 0
) -> float:
    """Compute the acquisition function for a given GP and a given point

    Args:
        x (np.ndarray): point to compute the acquisition function
        gp (GaussianProcess): GP to use for the computation
        a (float): tradeoff parameter. The higher the more exploration.
        weights (np.array): weights for the different tasks
        norm (float, optional): norm for the mean. Defaults to 2.
        c (float, optional): tradeoff parameter for the covariance. Defaults to 0.

    Returns:
        float: acquisition function value
    """
    if weights is None:
        weights = np.ones(gp.output_number)
    var = 0
    mean = 0
    covar = 0
    assert len(weights) == gp.output_number
    for i in range(gp.output_number):
        # Note that gp.input_dim is gp.output_number +1. It adds then the index of the task.
        x_pred = np.c_[x, np.zeros(x.shape[0]) + i].reshape(-1, gp.input_dim)
        var += (
            weights[i] * gp.posterior_covariance(x_pred, variance_only=c == 0)["v(x)"]
        )  # THIS IS THE VARIANCE!!!
        mean += weights[i] * gp.posterior_mean(x_pred)["f(x)"]
        if c != 0:
            # This is the covariance (between the tasks??).
            covar += weights[i] * gp.posterior_covariance(x_pred)["S(x)"][0]
    return norm * (mean + a * np.sqrt(var) + c * covar)
