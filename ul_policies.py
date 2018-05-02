from abc import ABCMeta, abstractmethod
from validation import check_random_state
from scaling import Scaling, NoScaling
from context_transformations import CONTEXT_TRANSFORMATIONS
import numpy as np

class UpperLevelPolicy(object):
    """Upper-level policy classes"""

    __metaclass__ = ABCMeta

    @abstractmethod
    def __call__(self, context=None, explore=True):
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

class BoundedScalingPolicy(UpperLevelPolicy):
    def __init__(self, upper_level_policy, scaling, bounds=None):
        self.upper_level_policy = upper_level_policy

        if scaling == "auto":
            if bounds is None:
                raise ValueError("scaling='auto' requires boundaries")
            else:
                covariance_diag = (bounds[:, 1] - bounds[:, 0]) ** 2 / 4.0
                scaling = Scaling(covariance=covariance_diag,
                                  compute_inverse=True)
        elif scaling == "none" or scaling is None:
            scaling = NoScaling()

        self.scaling = scaling
        self.bounds = bounds

    def transform_context(self, context):
        return self.upper_level_policy.transform_context(context)

    @property
    def W(self):
        return self.upper_level_policy.W

    @W.setter
    def W(self, W):
        self.upper_level_policy.W = W

    def __call__(self, context=None, explore=True):
        params = self.upper_level_policy(context, explore)
        params = self.scaling.scale(params)
        if self.bounds is not None:
            np.clip(params, self.bounds[:, 0], self.bounds[:, 1], out=params)

        return params

    def fit(self, X, Y, weights=None, context_transform=True):
        Y = self.scaling.inv_scale(Y.T).T
        self.upper_level_policy.fit(X, Y, weights, context_transform)

    def entroy(self):
        return self.upper_level_policy.entropy()

    @property
    def get_mean(self):
        return self.upper_level_policy.get_mean

    @property
    def get_sigma(self):
        return self.upper_level_policy.get_sigma

class ContextTransformationPolicy(UpperLevelPolicy):
    """
    A wrapper class around a policy which transform contexts
    """
    def __init__(self, PolicyClass, n_params, n_context_dims, context_transformation,
                 *args, **kwargs):
        """

        :param PolicyClass: subclass of upper level policy, the class of the actual policy,
                            which will be constructed internally
        :param n_params: int, dimensionality of weight vector of the low level policy
        :param n_context_dims: int, dimensionality of the context vector
        :param context_transformation: string or callable (Nonelinear) transformation for the context
        :param args:
        :param kwargs:
        """
        self.context_transformation = context_transformation
        if self.context_transformation is None:
            self.ct = CONTEXT_TRANSFORMATIONS["affine"]
        elif (isinstance(self.context_transformation, basestring) and
                      self.context_transformation in CONTEXT_TRANSFORMATIONS):
            self.ct = CONTEXT_TRANSFORMATIONS[self.context_transformation]
        else:
            self.ct = self.context_transformation

        self.n_features = self.transform_context(np.zeros(n_context_dims)).shape[0]

        # Create actual policy class to which all calls will be delegated after
        # context transformation
        self.policy = PolicyClass(n_params, self.n_features, *args, **kwargs)

    @property
    def W(self):
        return self.W

    @W.setter
    def W(self, W):
        self.policy.W = W


    def transform_context(self, context):
        return self.ct(context)

    def __call__(self, context=None, explore=True):
        context_features = self.transform_context(context)
        self.policy(context_features, explore)

    def fit(self, X, Y, weights=None, context_transform=True):
        if context_transform:
            X = np.array([self.transform_context(X[i]) for i in range(0, X.shape[0])])
        self.policy.fit(X, Y, weights)

    def entroy(self):
        return self.policy.entropy()

    @property
    def get_mean(self):
        return self.policy.get_mean

    @property
    def get_sigma(self):
        return self.policy.get_sigma

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

    @property
    def get_mean(self):
        return self.mean

    @property
    def get_sigma(self):
        return self.sigma

    def fit_quad_surrogate(self, eta, omega, F, f, context_transform=True):
        self.mean = np.dot(F, f)
        self.sigma = F * (eta + omega)


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

    def entroy(self):
        state_dim = self.mean.shape[0]
        _, logcov = np.linalg.slogdet(self.sigma)
        return 0.5 * (state_dim + state_dim * np.log(2 * np.pi) + logcov)


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

