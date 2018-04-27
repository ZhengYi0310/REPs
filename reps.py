from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from ul_policies import ConstantGaussianPolicy
from validation import check_random_state, check_feedback

def solve_dual_reps(R, epsilon, min_eta):
    """
    Solve the dual optimization function for reps
    :param R: array_like, (n_samples_per_update, ), obtained rewards
              from trajectories samples
    :param epsilon: float
                    Maximum Kullback-Leibler divergence of two successive policy
                    distributions.
    :param min_eta: Minimum eta, 0 would result in numerical problems
    :return: d : array, shape (n_samples_per_update,)
                Weights for training samples
                eta : float
                        Temperature
    """
    assert (R.ndim == 1, "Rewards must be passed in a flat array!")

    R_max = R.max()
    R_min = R.min()

    if R_max == R_min:
        return np.ones(R.shape[0]) / float(R.shape[0]), np.nan  # eta not known

    # Normalize returns into range [0, 1] such that eta (and min_eta)
    # always lives on the same scale
    R = (R - R_min) / (R_max - R_min)
    # Definition of the dual function
    def g(eta):
        return eta * epsilon + eta * np.log(np.sum(np.exp(R / eta)) / len(R))

    # Lower bounds for the lagrangian eta
    bounds = np.array([[min_eta, None]])

    x0 = np.array([1])
    r = fmin_l_bfgs_b(g, x0=x0, bounds=bounds, approx_grad=True)
    eta = r[0][0]

    # Determine weights of individual samples based on the their return and
    # the "temperature" eta
    log_d = (R / eta)
    d = np.exp(log_d - log_d.max())
    d /= d.sum()

    return d, r[0]


