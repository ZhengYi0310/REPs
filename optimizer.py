from abc import ABCMeta, abstractmethod
from base import Base

class Optimizer(Base):
    """
    Common interface for non-contextual optimizers
    """
    @abstractmethod
    def __init__(self, n_params):
        """
        Initialize the optimzer
        :param n_params: number of parameter vector
        """

    @abstractmethod
    def get_next_parameters(self, params):
        """
        Get the next individual/parameter vector for evaluations
        :param params: array_like, (n_params,)
        :return:
        """

    @abstractmethod
    def set_evaluation_feedback(self, rewards):
        """
        Set feedbacks of the parameter vector
        :param rewards: list of float
                        feedbacks for each step or for the episode, depends on the problem
        :return:
        """

    @abstractmethod
    def is_behavior_learning_done(self):
        """
        Check if the optimization is finished
        :return:
        """

    @abstractmethod
    def get_best_performance(self):
        """
        Get the best individual/parameter vector so far
        :return:
        """

class ContextualOptimizer(Base):
    """
    Common interface for (contextual) optimizers
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def init(self, n_params, n_contexts):
        """
        initializer optimizer

        :param n_params: int, dimension of the parameter vector
        :param n_contexts: dimension of the context space
        :return:
        """

    @abstractmethod
    def get_next_parameters(self, params):
        """
        Get the next individual/parameter vector for evaluations
        :param params: array_like, (n_params,)
        :return:
        """

    @abstractmethod
    def set_evaluation_feedback(self, rewards):
        """
        Set feedbacks of the parameter vector
        :param rewards: list of float
                        feedbacks for each step or for the episode, depends on the problem
        :return:
        """

    @abstractmethod
    def is_behavior_learning_done(self):
        """
        Check if the optimization is finished
        :return:
        """

    @abstractmethod
    def get_desired_context(self):
        """
        Choose the desired context for the next evaluation

        :return: context : ndarray-like, default=None
                 The context in which the next rollout shall be performed. If None,
                 the environment may select the next context without any preferences.
        """

    @abstractmethod
    def set_context(self, context):
        """
        Set context for next evaluation
        :param context: array-like, shape (n_contexts,)
                        The context in which the next rollout will be performed
        :return:
        """

    @abstractmethod
    def best_policy(self):
        """Return current best estimate of contextual policy.
        Returns
        -------
        policy : UpperLevelPolicy
            Best estimate of upper-level policy
        """
