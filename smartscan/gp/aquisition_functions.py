import numpy as np

def ndim_aqfunc(x, gp):
    """Compute the acquisition function for a given GP and a given point"""
    a = 2.5  #####change here, 3.0 for 95 percent confidence interval
    norm = 2.0
    ret = None
    for i in range(gp.input_dim-1):
        x_pred=np.c_[x,np.zeros(x.shape[0])+i].reshape(-1,gp.input_dim)
        cov = gp.posterior_covariance(x_pred)["v(x)"]
        if ret is None:
            ret = cov
        else:
            ret += cov
            
    ret = a * np.sqrt(ret)

    for i in range(gp.input_dim-1):
        x_pred=np.c_[x,np.zeros(x.shape[0])+i].reshape(-1,gp.input_dim)
        mean = gp.posterior_mean(x_pred)["f(x)"]
        ret += norm * mean

    return ret

