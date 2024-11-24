from maenvs4vrp.core.env_generator_builder import InstanceBuilder
from maenvs4vrp.core.env_observation_builder import ObservationBuilder
from maenvs4vrp.core.env_agent_selector import BaseSelector
from maenvs4vrp.core.env_agent_reward import RewardFn


from typing import Any, Dict, Iterable, Iterator, TypeVar, Tuple, Optional

import torch
from tensordict.tensordict import TensorDict

ObsType = TypeVar("ObsType")
ActionType = TypeVar("ActionType")
AgentID = str

ObsDict = Dict[AgentID, ObsType]
ActionDict = Dict[AgentID, ActionType]

class AECEnv():
    """
    Environment base class.     
    """

    DEFAULT_SEED = 2925
    def __init__(self,
            instance_generator_object: InstanceBuilder,  
            obs_builder_object: ObservationBuilder,
            agent_selector_object: BaseSelector,
            reward_evaluator: RewardFn,
            seed:int = None,               
            device: Optional[str] = None,
            batch_size: torch.Size = None,
            ):        
        """
        Constructor

        Args:
            instance_generator_object(InstanceBuilder): Generator instance.
            obs_builder_object(ObservationBuilder): Observations instance.
            agent_selector_object(BaseSelector): Agent selector instance
            reward_evaluator(RewardFn): Reward evaluator instance.
            seed(int): Random number generator seed. Defaults to None.
            device(str, optional): Type of processing. It can be "cpu" or "gpu". Defaults to None.
            batch_size(torch.Size): Batch size. Defaults to None.

        """

        if seed is None:
            self._set_seed(self.DEFAULT_SEED)
        else:
            self._set_seed(seed)

        self.agent_selector = agent_selector_object

        self.inst_generator = instance_generator_object
        self.inst_generator._set_seed(self.seed)
        self.obs_builder = obs_builder_object
        self.obs_builder.set_env(self)      
        self.reward_evaluator = reward_evaluator
        self.reward_evaluator.set_env(self)   
              
        if device == None:
            self.device = instance_generator_object.device
        else:
            self.device = device

        if batch_size is None:
            self.batch_size =  self.inst_generator.batch_size
        else:
            batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
            self.batch_size = torch.Size(batch_size)
            instance_generator_object.batch_size = torch.Size(batch_size)


        self.num_nodes = self.inst_generator.max_num_nodes
        self.num_agents = self.inst_generator.max_num_agents

        self.nodes_static_feat_dim = self.obs_builder.get_nodes_static_feat_dim()
        self.nodes_dynamic_feat_dim = self.obs_builder.get_nodes_dynamic_feat_dim()
        self.agent_feat_dim = self.obs_builder.get_agent_feat_dim()
        self.agents_feat_dim = self.obs_builder.get_other_agents_feat_dim()
        self.global_feat_dim = self.obs_builder.get_global_feat_dim()


    def _set_seed(self, seed: Optional[int]):
        """
        Set the random seed used by the environment.
        
        Args:
            seed(int, optional): Seed to be set.

        Returns:
            None.
        """
        self.seed = seed
        rng = torch.manual_seed(self.seed)
        self.rng = rng


    def observe(self, agent_name:AgentID)-> TensorDict:
        """
        Compute the environment.

        Args:
            agent_name(AgentID): Current agent.

        Returns
            TensorDict: Current agent observaions and masks dictionary.
        """
        raise NotImplementedError()


    def sample_action(self, agent_name:AgentID)-> TensorDict:
        """
        Compute a random action from avaliable actions to current agent.
        
        Args:
            agent_name(AgentID): Current agent.

        Returns:
            TensorDict: Tensor environment instance with updated action.
        """
        raise NotImplementedError()

    def reset(self) -> TensorDict:
        """
        Reset the environment to a starting state and return infos dict.

        Args:
            n/a.

        Returns:
            TensorDict: Environment information.
        """
        raise NotImplementedError()


    def step(self, action:ActionType)-> TensorDict:
        """
        Perform an environment step for active agent.

        Args:
            action(ActionType): Action to perform.

        Returns:
            TensorDict: Updated tensor environment instance.

        """
        raise NotImplementedError()
