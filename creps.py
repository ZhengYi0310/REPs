import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from optimizer import ContextualOptimizer
from scaling import Scaling, NoScaling
from ul_policies import LinearGaussianPolicy, BoundedScalingPolicy, ContextTransformationPolicy
from validation import check_random_state, check_feedback, check_context
from log import  get_logger
from collections import deque

def solve_dual_contextual_reps(S, R, epsilon, min_eta):
    """
    Solve the dual optimization problem for the contextual reps
    :param S: array_like, array, shape (n_samples_per_update, n_context_features)
              Features for the context-dependend reward baseline
    :param R: array_like, (n_samples_per_update, ), obtained rewards
              from trajectories samples
    :param epsilon: float
                    Maximum Kullback-Leibler divergence of two successive policy
                    distributions.
    :param min_eta:Minimum eta, 0 would result in numerical problems
    :return:
    """
    if S.shape[0] != R.shape[0]:
        raise ValueError("Number of contexts (%d) must equal number of "
                         "returns (%d)." % (S.shape[0], R.shape[0]))

    n_samples_per_update = R.shape[0]

    # definition of the dual function
    def g(x):
        eta = x[0]
        nu = x[1:]
        return eta * epsilon + np.dot(nu.T, np.mean(S, axis=0)) + eta * np.log(np.sum(np.exp(R - np.dot(nu.T, S)) / eta)
                                                                               / n_samples_per_update)

    bounds = np.vstack(([[min_eta, None]], np.tile(None, (S.shape[1], 2))))
    x0 = np.array([1] + [1] * S.shape[1])
    r = fmin_l_bfgs_b(g, x0, approx_grad=True, bounds=bounds)

    eta = r[0][0]
    nu = r[0][1:]

    log_d = (R - np.dot(nu.T, S)) / eta
    d = np.exp(log_d - log_d.max())
    return d, eta, nu

class CREPSOptimizer(ContextualOptimizer):
    """
    Contextual Relative Entropy Policy Search
    Use C-REPS as a black-box contextual optimizer: Learns an upper-level
    distribution :math:`\pi(\\boldsymbol{\\theta}|\\boldsymbol{s})` which
    selects weights :math:`\\boldsymbol{\\theta}` for the objective function.
    At the moment, :math:`\pi(\\boldsymbol{\\theta}|\\boldsymbol{s})` is
    assumed to be a multivariate Gaussian distribution whose mean is a linear
    function of nonlinear features from the context. C-REPS constrains the
    learning updates such that the KL divergence between successive
    distribution is below the threshold :math:`\epsilon`.

    References
    ----------
    .. [1] Kupcsik, A.; Deisenroth, M.P.; Peters, J.; Loh, A.P.;
        Vadakkepat, P.; Neumann, G. Model-based contextual policy search for
        data-efficient generalization of robot skills.
        Artificial Intelligence 247, 2017.
    """
    def __init__(self, initial_params=None, variance=None, covariance=None,
                 epsilon=2.0, min_eta=1e-8, train_freq=25,
                 n_samples_per_update=100, context_features=None, gamma=1e-4,
                 bounds=None, log_to_file=False, log_to_stdout=False, random_state=None,
                 **kwargs):
        """

        :param initial_params: array-like, shape (n_params, )
        :param variance: float, initial exploration variance
        :param covariance: array-like, optional (default: None)
                           Either a diagonal (with shape (n_params,)) or a full covariance matrix
                           (with shape (n_params, n_params)). A full covariance can contain
                           information about the correlation of variables.
        :param epsilon: float, optional (default: 2.0)
                        Maximum Kullback-Leibler divergence of two successive policy
                        distributions.
        :param min_eta: float, optional (default: 1e-8)
                        Minimum eta, 0 would result in numerical problems
        :param train_freq:  int, optional (default: 25)
                            Number of rollouts between policy updates.
        :param n_samples_per_update: int, optional (default: 100)
                                     Number of samples that will be used to update a policy.
        :param context_features: string or callable, optional (default: None)
                                 (Nonlinear) feature transformation for the context. Possible options
                                 are 'constant', 'linear', 'affine', 'quadratic', 'cubic', or you can
                                 write a custom function that computes a transformation of the context.
                                 This will make a linear upper level policy capable of representing
                                 nonlinear functions.
        :param gamma: float, optional (default: 1e-4)
                      Regularization parameter. Should be removed in the future.
        :param bounds: array-like, shape (n_samples, 2), optional (default: None)
                       Upper and lower bounds for each parameter.
        :param log_to_file: optional, boolean or string (default: False)
                            Log results to given file, it will be located in the $BL_LOG_PATH
        :param log_to_stdout: optional, boolean (default: False)
                              Log to standard output
        :param random_state: optional, int
                             Seed for the random number generator.
        :param kwargs:
        """
        self.initial_params = initial_params
        self.variance = variance
        self.covariance = covariance
        self.epsilon = epsilon
        self.min_eta = min_eta
        self.train_freq = train_freq
        self.n_samples_per_update = n_samples_per_update
        self.context_features = context_features
        self.gamma = gamma
        self.bounds = bounds
        self.log_to_file = log_to_file
        self.log_to_stdout = log_to_stdout
        self.random_state = random_state

    def init(self, n_params, n_context_dims):
        """
        Initialize the optimizer
        :param n_params: number of parameters
        :param n_context_dims: the dimensionality of the context features
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
        self.context = None
        self.params = None
        self.reward = None

        scaling = Scaling(variance=self.variance, covariance=self.covariance,
                          compute_inverse=True)
        inv_scaled_params = scaling.inv_scale(self.initial_params)
        policy = ContextTransformationPolicy(LinearGaussianPolicy, n_params, n_context_dims,
                                             context_transformation=self.context_features,
                                             mean = inv_scaled_params, covariance_scale=1.0,
                                             gamme=self.gamma, random_state=self.random_state)
        self.policy_ = BoundedScalingPolicy(policy, scaling, self.bounds)

        # Maximum return obtained
        self.max_return = -np.inf
        # Best parameters found so far
        self.best_params = self.initial_params.copy()

        self.history_theta = deque(maxlen=self.n_samples_per_update)
        self.history_R = deque(maxlen=self.n_samples_per_update)
        self.history_s = deque(maxlen=self.n_samples_per_update)
        self.history_phi_s = deque(maxlen=self.n_samples_per_update)

    def get_desired_context(self):
        """C-REPS does not actively select the context.
        Returns
        -------
        context : None
            C-REPS does not have any preference
        """
        return None

    def set_context(self, context):
        """Set context of next evaluation.
            Parameters
            ----------
            context : array-like, shape (n_context_dims,)
                The context in which the next rollout will be performed
        """
        self.context = check_context(context)

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

        params[:] = self.params

    def set_evaluation_feedback(self, rewards):
        self.reward = check_feedback(self.reward, compute_sum=True)
        if self.log_to_stdout or self.log_to_file:
            self.logger.info("[CREPS] Reward %.6f" % self.reward)

        phi_s = self.policy_.transform_context(self.context)

        self.history_theta.append(self.params)
        self.history_R.append(self.reward)
        self.history_phi_s.append(phi_s)
        self.history_s.append(self.context)

        self.it += 1

        if self.it % self.train_freq == 0:
            theta = np.asarray(self.history_theta)
            R = np.asarray(self.history_R)
            phi_s = np.asarray(self.history_phi_s)

            d = solve_dual_contextual_reps(phi_s, R, self.epsilon, self.min_eta)[0]
            self.policy_.fit(phi_s, theta, d)

    def best_policy(self):
            """Return current best estimate of contextual policy.
            Returns
            -------
            policy : UpperLevelPolicy
                Best estimate of upper-level policy
            """
            return self.policy_

    def is_behavior_learning_done(self):
            """Check if the optimization is finished.
            Returns
            -------
            finished : bool
                Is the learning of a behavior finished?
            """
            return False

    def __getstate__(self):
            d = dict(self.__dict__)
            del d["logger"]
            return d

    def __setstate__(self, d):
            self.__dict__.update(d)
            self.logger = get_logger(self, self.log_to_file, self.log_to_stdout)