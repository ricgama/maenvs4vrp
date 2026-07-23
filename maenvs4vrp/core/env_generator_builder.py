
from typing import Dict, Optional
import random
import numpy as np
import torch
from tensordict.tensordict import TensorDict
from wandb.util import np


class InstanceBuilder(object):

    """
    Instance generator base class.
    """


    DEFAULT_SEED = 2925
    def __init__(self, instance_name:Optional[str]=None,
                 list_of_instances:Optional[set]=None,
                 num_nodes:Optional[int]=None,
                 num_agents:Optional[int]=None,
                 seed:Optional[int]=None,
                 device: str = "cpu",
                 batch_size: torch.Size = None) -> None:
        """
        Constructor

        Args:
            instance_name(str): Instance name. Defaults to None.
            list_of_instances(set):  List of instances file names. Defaults to None.
            num_nodes(int):  Total number of nodes. Defaults to None.
            num_agents(int):  Total number of agents. Defaults to None.
            seed(int): Random number generator seed. Defaults to None.
            device (str): Type of processing. Defaults to "cpu".
            batch_size(torch.Size or None): Batch size. If not specified, defaults to 1.
        """

        self.num_nodes = None
        self.num_agents = None
        # seed the generation process
        if seed is None:
            self._set_seed(self.DEFAULT_SEED)
        else:
            self._set_seed(seed)

        self.device = device
        if batch_size is None:
            batch_size = [1]
        else:
            batch_size = [batch_size] if isinstance(batch_size, int) else batch_size

        self.batch_size = torch.Size(batch_size)

    def _set_seed(self, seed: Optional[int]):
        """
        Set the random seed used by the environment.

        Args:
            seed(int, optional): Seed used.

        Returns:
            None.
        """
        self.seed = seed
        # 1. Python built-in random module
        random.seed(seed)
        # 2. NumPy library
        np.random.seed(seed)
        # 3. PyTorch (CPU and all GPUs)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def read_instance_data(self, instance_name:str)-> Dict:
        """
        Read instance data from file.

        Args:
            instance_name(str): instance file name.

        Returns:
            Dict: Instance data.
        """
        raise NotImplementedError()

    def get_instance(self, instance_name:str,
                     num_agents:Optional[int]=None)-> Dict:
        """
        Combine read instance file and parse to Dict.

        Args:
            instance_name(str): Instance file name.
            num_agents(int): Number of agents. Defaults to None.

        Returns:
            Dict: Instance data.
        """
        raise NotImplementedError()

    def load_list_of_instances(self, set_of_instances: Optional[set] = None,
                               already_loaded: Optional[bool] = None):
        """
        Load every instance on set_of_instances set.

        Args:
            set_of_instances(Optional[set], optional): Set of instances file names. Defaults to None.
            already_loaded(Optional[bool], optional): If instance data has been pre-loaded. Defaults to None.

        """
        raise NotImplementedError()

    def get_instance_preloaded(self) -> Dict:
        """
        Get preloaded instance.

        Args:
            n/a.

        Returns:
            Dict: Instance data.
        """
        raise NotImplementedError()



    def random_sample_instance(self, num_agents: Optional[int] = None,
                               num_nodes: Optional[int] = None,
                               seed: Optional[int] = None) -> Dict:
        """
        Sample one instance from instance space.

        Args:
            num_nodes(int):  Total number of nodes. Defaults to None.
            num_agents(int):  Total number of agents. Defaults to None.
            seed(int): Random number generator seed. Defaults to None.

        Returns:
            Dict: Instance data.
        """
        raise NotImplementedError()

    def sample_name_from_list(self, seed: Optional[int] = None) -> str:
        """
        Sample one instance from instance list.

        Args:
            seed(int, optional): Random number generator seed. Defaults to None.

        Returns:
            str: instance name.
        """
        raise NotImplementedError()

    def sample_instance(self, num_agents: Optional[int] = None,
                        num_nodes: Optional[int] = None,
                        instance_name: Optional[str] = None,
                        random_sample: bool = True,
                        seed: Optional[int] = None) -> Dict:
        """
        Sample one instance from instance space.

        Args:
            num_nodes(int, optional): Total number of nodes. Defaults to None.
            num_agents(int, optional): Total number of agents. Defaults to None.
            instance_name(str, optional): Instance name. Defaults to None.
            random_sample(bool, optional): True to sample instance and False to use original instance data. Defaults to None.
            seed(int, optional): Random number generator seed. Defaults to None.

        Returns:
            Dict: Instance data.
        """
        raise NotImplementedError()
