import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from optimizer import Optimizer
from scaling import Scaling, NoScaling
from ul_policies import ConstantGaussianPolicy, BoundedScalingPolicy
from validation import check_random_state, check_feedback, check_context
from log import  get_logger
from collections import deque
from sklearn.preprocessing import PolynomialFeatures

def fit_quadratic_surrogate(rewards, params):
    """
    Fit a quafratice surrogate function which relates the low level policy parameters (DMP weights) with reward
    :param R: (n_samples_per_update, ), obtained rewards from trajectories samples
    :param params: (n_samples_per_update, params_dim)
    :return:
    """
    states_dim = params.shape[1]
    states_feats = PolynomialFeatures(2).transform(params)

    # Fit the beta
    temp1 = np.linalg.pinv(np.dot(states_feats.T, states_feats) + 1e-8 * np.eye(states_feats.shape[1]))
    temp2 = np.dot(states_feats.T, rewards)
    Beta = np.dot(temp1, temp2)

    triu_indices = np.triu_indices(states_dim, k=0)
    tril_indices = np.tril_indices(states_dim, k=0)

    R = np.zeros(states_dim, states_dim)
    R[triu_indices] = Beta[1 + states_dim: ]
    R[tril_indices] = R.T[tril_indices]

    triu_indices = np.triu_indices(states_dim, k=1)
    tril_indices = np.tril_indices(states_dim, k=-1)
    R[triu_indices] = R[triu_indices] / 2
    R[tril_indices] = R[tril_indices] / 2

    # check for negative definitness
    w, m = np.linalg.eig(R)
    w[w >= 0] = -1e-6

    R = np.dot(m, np.dot(np.diag(w), m.T))
    residual = rewards - np.dot(params.T, np.dot(R, params))
    states_feats = PolynomialFeatures(1).transform(params)
    temp1 = np.linalg.pinv(np.dot(states_feats.T, states_feats) + 1e-8 * np.eye(states_feats.shape[1]))
    temp2 = np.dot(states_feats.T, residual)
    Beta = np.dot(temp1, temp2)
    r0 = Beta[0]
    r = Beta[1: ]

    return R, r, r0

def solve_dual_reps_moreps(policy_mean, policy_variance, R, r0, r, epsilon, beta, min_eta):

    b = policy_mean
    inv_Q = np.linalg.inv(policy_variance)
    def g(x, *args):
        eta = x[0]
        omega = x[1]
        b = args[0]
        inv_Q = args[0]

        F = np.linalg.inv(eta * inv_Q - 2 * R)
        f = eta * np.dot(inv_Q, b) + r

        _, temp_logdet1 = np.linalg.slogdet(2 * np.pi * policy_variance)
        _, temp_logdet2 = np.linalg.slogdet(2 * np.pi * (eta + omega) * F)

        func =  eta * epsilon -  beta * omega + 0.5 * (np.dot(f.T, np.dot(F, f)) -
                                                       eta * np.dot(b.T, np.dot(inv_Q, b)) -
                                                       eta * temp_logdet1 + (eta + omega) * temp_logdet2)

        dF_deta = -np.dot(F, inv_Q, F)
        df_deta = np.dot(policy_variance, policy_mean)

        eta_prime = epsilon + 0.5 * (np.dot(f.T, np.dot(dF_deta, f)) + 2 * np.dot(f.T, np.dot(F, df_deta)) -
                                     np.dot(b.T, np.dot(inv_Q, b)) - temp_logdet1 + temp_logdet2 + R.shape[0] -
                                     (eta + omega) * np.trace(np.dot(inv_Q, F)))

        omega_prime = -beta + 0.5 * (temp_logdet2 + R.shape[0])

        return func, np.array([eta_prime, omega_prime])

    bounds = np.vstack(([[min_eta, 1e8]], [1e-8, 1e8]))
    x0 = np.array([0.5, 0.5])
    results = fmin_l_bfgs_b(g, x0=x0, bounds=bounds, args=(b, inv_Q))

    eta = results[0][0]
    omega = results[0][1]

    F = np.linalg.inv(eta * inv_Q - 2 * R)
    f = eta * np.dot(inv_Q, b) + r

    return eta, omega, F, f


class MOREPSOptimizer(Optimizer):
    """
    model-based replative entrpy stochastic search
    References
    ----------
    .. [1] Abdolmaleki, A.; Lioutikov, R.; Lau, N.; Reis, L.P; Peters, J.; Neumann, G. Model-Based Relative Entropy Stochastic Search.
        Advances in Neural Information Processing Systems, 2015.
    """

    def __init__(self, initial_params=None, variance=1.0, covariance=None, epsilon=2.0, gamma=0.99, ini_entropy=75, min_eta=1e-8,
                 train_freq=25, n_samples_per_update=100, bounds=None, log_to_file=False,
                 log_to_stdout=False, random_state=None):
        self.initial_params = initial_params
        self.variance = variance
        self.covariance = covariance
        self.epsilon = epsilon
        self.min_eta = min_eta
        self.train_freq = train_freq
        self.n_samples_per_update = n_samples_per_update
        self.bounds = bounds
        self.log_to_file = log_to_file
        self.log_to_stdout = log_to_stdout
        self.random_state = random_state
        self.ini_entropy = ini_entropy
        self.gamma = gamma

    def init(self, n_params):
        """
        Initialize the optimizer
        :param n_params: number of parameters
        :return:
        """
        self.logger = get_logger(self, self.log_to_file, self.log_to_stdout)
        self.random_state = check_random_state(self.random_state)

        self.it = 0

        if self.initial_params == None:
            self.initial_params = np.zeros(n_params)
        else:
            self.initial_params = np.asarray(self.initial_params).astype(np.float64, copy=True)

        if n_params != len(self.initial_params):
            raise ValueError("Number of dimensions {} does not match "
                             "number of initial parameters {}.".format(n_params,
                                                                       len(self.initial_params)))

        self.params = None
        self.reward = None

        scaling = Scaling(variance=self.variance, covariance=self.covariance,
                          compute_inverse=True)
        self.policy_ = BoundedScalingPolicy(ConstantGaussianPolicy(
            n_params, mean=scaling.inv_scale(self.initial_params),
            random_state=self.random_state), scaling=scaling,
            bounds=self.bounds)

        # Maximum return obtained
        self.max_return = -np.inf
        # Best parameters found so far
        self.best_params = self.initial_params.copy()

        self.history_theta = deque(maxlen=self.n_samples_per_update)
        self.history_R = deque(maxlen=self.n_samples_per_update)

    def get_next_parameters(self, params, explore=True):
        """Return parameter vector that shall be evaluated next.
            Parameters
            ----------
            params : array-like, shape = (n_params,)
                The selected parameters will be written into this as a side-effect.
            explore : bool
                Whether exploration in parameter selection is enabled
            """
        self.params = self.policy_(None, explore=explore)

    def set_evaluation_feedback(self, rewards):
        """
        Inform the optimizer of the outcome of a rollout with current weights
        :param rewards:
        :return:
        """
        self.reward = check_feedback(rewards, compute_sum=True)
        self.history_theta.append(self.params)
        self.history_R.append(self.reward)

        self.it += 1
        if self.it % self.train_freq == 0:
            theta = np.asarray(self.history_theta)
            R = np.asarray(self.history_R)
            R, r, r0 = fit_quadratic_surrogate(R, theta)
            beta = self.gamma * (self.policy_.entroy() - self.ini_entropy) * self.ini_entropy
            eta, omega, F, f = solve_dual_reps_moreps(self.policy_.get_mean, self.policy_.get_sigma, R, r0, r, self.epsilon, beta, self.min_eta)
            self.policy_.upper_level_policy.fit_quad_surrogate(eta, omega, F, f)

        self.logger.info("Reward %.6f" % self.reward)

        if self.reward > self.max_return:
            self.max_return = self.reward
            self.best_params = self.params

    def get_best_parameters(self):
        """Get the best parameters.
            Returns
            -------
            best_params : array-like, shape (n_params,)
                Best parameters
        """

        return self.best_params

    def is_behavior_learning_done(self):
            """Check if the optimization is finished.
            Returns
            -------
            finished : bool
                Is the learning of a behavior finished?
            """

            return False














