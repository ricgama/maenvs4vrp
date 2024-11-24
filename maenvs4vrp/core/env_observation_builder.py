from typing import Optional, Dict, List


class ObservationBuilder:
    """Observations base class.
    """

    POSSIBLE_NODES_STATIC_FEATURES:List[str] = []
    POSSIBLE_NODES_DYNAMIC_FEATURES:List[str] = []
    POSSIBLE_AGENT_FEATURES:List[str] = []
    POSSIBLE_OTHER_AGENTS_FEATURES:List[str] = []
    POSSIBLE_GLOBAL_FEATURES:List[str] = []

    def __init__(self, feature_list:Dict = None):
        """
        Constructor

        Args:
            feature_list(Dict): Dictionary containing observation features list to be available to the agent. Defaults to None.

        """
        self.env = None
        self.default_feature_list:Dict = {}

        self.possible_nodes_static_features = self.POSSIBLE_NODES_STATIC_FEATURES
        self.possible_nodes_dynamic_features = self.POSSIBLE_NODES_DYNAMIC_FEATURES
        self.possible_agent_features = self.POSSIBLE_AGENT_FEATURES
        self.possible_agents_features = self.POSSIBLE_OTHER_AGENTS_FEATURES
        self.possible_global_features = self.POSSIBLE_GLOBAL_FEATURES

    def set_env(self, env):

        """
        Set environment.

        Args:
            env(AECEnv): Environment.

        Returns:
            None.
        """

        self.env = env

    def get_static_feat_dim(self):
        """
        Get nodes static features dimensions.

        Args:
            n/a.

        Returns:
            int: Nodes static features dimensions.
        """
        raise NotImplementedError()

    def get_dynamic_feat_dim(self):
        """
        Get nodes dynamic features dimensions.

        Args:
            n/a.

        Returns:
            int: Nodes dynamic features dimensions.
        """
        raise NotImplementedError()

    def get_nodes_feat_dim(self):
        """
        Get nodes features dimensions.

        Args:
            n/a.

        Returns:
            int: Nodes features dimensions.
        """
        raise NotImplementedError()

    def get_agent_feat_dim(self):
        """
        Get agent features dimensions.

        Args:
            n/a.

        Returns:
            int: Agent features dimensions.
        """
        raise NotImplementedError()

    def get_other_agents_feat_dim(self):
        """
        Get other agent features dimensions.

        Args:
            n/a.

        Returns:
            int: Other agent features dimensions.
        """
        raise NotImplementedError()

    def get_global_feat_dim(self):
        """
        Get global features dimensions.

        Args:
            n/a.

        Returns:
            int: Global features dimensions.
        """
        raise NotImplementedError()

    def compute_static_features(self):
        """
        Get nodes static features.

        Args:
            n/a.

        Returns:
            torch.Tensor: Nodes static features.
        """
        raise NotImplementedError()

    def compute_dynamic_features(self):
        """
        Get nodes dynamic features.

        Args:
            n/a.

        Returns:
            torch.Tensor: Nodes dynamic features.
        """
        raise NotImplementedError()

    def compute_agent_features(self):
        """
        Get current agent features.

        Args:
            n/a.

        Returns:
            torch.Tensor: Current agent features.
        """
        raise NotImplementedError()

    def compute_agents_features(self):
        """
        Get other agent features.

        Args:
            n/a.

        Returns:
            torch.Tensor: Other agent features.
        """
        raise NotImplementedError()

    def compute_global_features(self):
        """
        Get global features.

        Args:
            n/a.

        Returns:
            torch.Tensor: Global features.
        """
        raise NotImplementedError()

    def get_observations(self):
        """
        Compute the environment.

        Args:
            n/a.

        Returns
            observations(TensorDict): Current environment observations and masks dictionary.
        """
        raise NotImplementedError()


