from maenvs4vrp.core.env_generator_builder import InstanceBuilder
from maenvs4vrp.core.env_observation_builder import ObservationBuilder
from maenvs4vrp.core.env_agent_selector import BaseSelector
from maenvs4vrp.core.env_agent_reward import RewardFn


from typing import Any, Dict, Iterable, Iterator, TypeVar, Tuple, Optional

import torch
from tensordict.tensordict import TensorDict

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
            batch_size: Optional[torch.Size] = None,
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


    def observe(self, is_reset=False)-> TensorDict:
        """
        Compute the environment.

        Args:
            is_reset(bool): If the environment is on reset. Defauts to False.

        Returns
            TensorDict: Current agent observaions and masks dictionary.
        """
        raise NotImplementedError()


    def sample_action(self, td: TensorDict)-> TensorDict:
        """
        Compute a random action from avaliable actions to current agent.
        
        Args:
            td(TensorDict): Environment instance tensor.

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


    def step(self, td: TensorDict) -> TensorDict:
        """
        Perform an environment step for active agent.

        Args:
            td(TensorDict): Environment tensor instance.

        Returns:
            TensorDict: Updated tensor environment instance.

        """
        raise NotImplementedError()
    
    def _get_current_instance_data(self) -> dict[str, dict[str, torch.Tensor] | str]:
        """
        Return a lightweight, plotting-friendly view of the current instance.

        Expected to include at least:
          - 'data': a dict with tensors such as 'coords' (shape [B, N, 2] or [N, 2]),
                    'is_depot' or 'depot_idx' if available, and any other fields
                    listed in self.instance_data_keys.
          - 'name': a human-friendly instance name.

        This structure is consumed by plotting utilities like plot_instance_coords.
        """
        instance = {"name": getattr(self, "instance_name", f"{self.env_name.upper()}-Instance"), "data": {}}
        keys = getattr(self, "instance_data_keys", None)

        # If instance_data_keys is not defined, fall back to common fields.
        if keys is None:
            keys = []
            for k in ("coords", "is_depot", "depot_idx"):
                if k in self.td_state:
                    keys.append(k)

        for k in keys:
            if k in self.td_state:
                instance["data"][k] = self.td_state[k]
        return instance
