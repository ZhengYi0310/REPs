from abc import ABCMeta, abstractmethod
from validation import check_random_state
import numpy as np

class UpperLevelPolicy(object):
    """Upper-level policy classes"""

    __metaclass__ = ABCMeta

    @abstractmethod
    def __call__(self, context=None, explore=None):
        """
        Policy evaluation

        Sample weight vector from distribution if explor is true,
        otherwise return the mean of the distribution
        :param context:  array-like, shape (n_context_dims,), optional (default: None)
                         Context vector (ignored by non-contextual policies)
        :param explore: if true, weight vector is sampled from distribution. otherwise the
                        distribution's mean is returned
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def fit(self, X, Y, weights, context_transform=True):
        """
        Train policy by maximum likelihood estimation
        :param X: array-like, shape (n_samples, n_context_dims)
                  2d array of context vectors
        :param Y: array-like, shape (n_samples, n_params)
                  2d array of parameter vectors
        :param weights: array-like, (n_samples,)
                        weights of individual samples (weight vectors)
        :param context_transform:
        :return:
        """
        raise NotImplementedError

    def transform_context(self, context):
        """Transform context based on internal context transformation. """
        return context  # no transformation as default


class ConstantGaussianPolicy(UpperLevelPolicy):
    """
    Gaussian policy with a constant mean
    upper-level policy, which samples weight vectors for low level

    with constant mean mu and covariance Sigma. Thus, contextual information
    cannot be taken into account

    see [A Survey A Survey on Policy Search for Robotics, p.131]
    """
    def __init__(self, n_weights, covairance="full", mean=None,
                 covariance_scale=1.0, random_state=None):
        """

        :param n_weights: dimensionality of the weight for the low level policies
                          (such as DMP)
        :param covairance: string ("full" or "diag") whether full or diagonal convariance
                           matrices is learned
        :param mean: array-like, shape (n_weights)
                     initial mean of policy
        :param covariance_scale: float
                                 the covariance is initialized to numpy.eye(n_weights) *
                                 covariance_scale
        :param random_state: random number generator for
        """

        self.n_weights = n_weights
        self.covariance = covairance
        self.random_state = check_random_state(random_state)

        self.mean = np.ones(n_weights) if (self.mean == None) else mean
        self.sigma = np.eye(n_weights) * covariance_scale

    def __call__(self, context=None, explore=None):

        if explore is None:
            return self.mean
        else:
            return self.random_state.multivariate_normal(self.mean, self.sigma, size=1)[0]

    def fit(self, X, Y, weights, context_transform=True):
        """

        :param X: ignored
        :param Y: array-like, shape (num_samples, n_weights)
                  2d array of parameter vectors
        :param weights: array-like, (n_samples,)
                        weights of individual samples (weight vectors)
        :param context_transform: ignored
        :return:
        """
        weights[weights==0] = np.finfo(float).eps

        self.mean = np.sum(weights * Y.T, axis=1) / np.sum(weights)

        # Estimate covariance matrix (either full or diagonal)
        Z = (np.sum(weights) ** 2 - np.sum(weights ** 2)) / np.sum(weights)

        if self.covariance == "full":
            temp = Y - self.mean
            nominator = np.zeros_like(self.sigma)
            for i in range(0, len(weights)):
                nominator += weights[i] * np.outer(temp[i, :], temp[i, :])
            self.sigma = nominator / (1e-10 + Z)
        elif self.covariance == "diag":
            nominator = np.zeros(Y.shape[0])
            temp = (Y - self.mean) ** 2
            nominator = np.sum(weights * temp.T, axis=1)
            self.sigma = np.diag(nominator / (1e-10 + Z))

        if not np.isfinite(self.sigma).all():
            raise ValueError("covairance matrice has non-finite terms")

    def likehood(self, Y):
        """
        Likelihood of observations vector Y
        :param Y:observation vector (n_samples, n_weights)
        :return:
        """
        temp = -0.5 * np.dot(np.dot(Y - self.mean, np.linalg.inv(self.sigma)), Y - self.mean)
        temp = np.exp(np.diag(temp))
        temp = temp / np.sqrt(np.linalg.det(np.pi * 2 * self.sigma))
        return temp

class LinearGaussianPolicy(UpperLevelPolicy):
    """
    Gaussian policy with mean that is linear in state feature/contexts

    The distribution's mean depends linearly on the context x via the matrix W
    but the covariance Sigma is context independent.

    see [A Survey A Survey on Policy Search for Robotics, p.131]

    """

    def __init__(self, n_weights, n_context, mean=None,
                 covariance_scale=1.0, gamma=0.0, random_state=None):
        """

        :param n_weights: dimensionality of the weight for the low level policies
                          (such as DMP)
        :param n_context: dimensionality of the context/state feature
        :param mean: array-like, shape (n_weights)
                     initial mean of policy
        :param gamma: float, optional (default: 0)
                      regularization parameter for weighted maximum likelihood estimation
                      of W.
        :param covariance_scale: float
                                 the covariance is initialized to numpy.eye(n_weights) *
                                 covariance_scale
        :param random_state:
        """
        self.n_weights = n_weights
        self.n_context = n_context
        self.covariance_scale = covariance_scale
        self.gamma = gamma
        self.random_state = check_random_state(random_state)

        self.W = np.zeros((n_weights, n_context))
        if self.mean is not None:
            # It is assumed that the last dimension of the context is a
            # constant bias dimension
            self.W[:, -1] = self.mean
        self.mean = np.ones(n_weights) if (self.mean == None) else mean
        self.sigma = np.eye(n_weights) * covariance_scale

    def __call__(self, context=None, explore=None):
        """
        Evalute policy for the given context
        Samples weight vector from distribution if explore is true, otherwise
        return the distribution's mean (which depends on the context).

        :param context: context or state feature
        :param explore:
        :return:
        """

        if explore is None:
            return np.dot(self.W, context)
        else:
            return self.random_state.multivariate_normal(self.mean, self.sigma, size=1)[0]

    def fit(self, X, Y, weights, context_transform=True):
        """

        :param X: array-like, shape (num_samples, n_contexts)
                  2d array of parameter vectors
        :param Y: array-like, shape (num_samples, n_weights)
                  2d array of parameter vectors
        :param weights: array-like, (n_samples,)
                        weights of individual samples (weight vectors)
        :param context_transform: ignored
        :return:
        """
        weights[weights == 0] = np.finfo(float).eps

        self.mean = np.sum(weights * Y.T, axis=1) / np.sum(weights)

        # Estimate covariance matrix (either full or diagonal)
        Z = (np.sum(weights) ** 2 - np.sum(weights ** 2)) / np.sum(weights)

        D = np.diag(weights)
        temp1 = np.linalg.pinv(np.dot(np.dot(X.T, D), X) + np.eye(X.shape[1]))
        temp2 = np.dot(X.T, np.dot(D, Y))
        self.W = np.dot(temp1, temp2).T

        nominator = np.zeros_like(self.sigma)
        for i in range(0, len(weights)):
            temp = np.dot(self.W, X[i, :])
            nominator += weights[i] * np.outer(temp, temp)
        self.sigma = nominator / (1e-10 + Z)

