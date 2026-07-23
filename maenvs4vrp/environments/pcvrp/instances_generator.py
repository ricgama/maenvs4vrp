import torch
from tensordict import TensorDict

import os
from os import path
import pickle

from typing import Dict, Optional
from maenvs4vrp.core.env_generator_builder import InstanceBuilder

import warnings
import random

from huggingface_hub import HfApi, snapshot_download
import shutil

HF_REPO_ID = "maenvs4vrp/environments"
INSTANCES_PATH = 'pcvrp/data/generated'
DATA_PATH = './pcvrp/data/generated'

class InstanceGenerator(InstanceBuilder):
    """
    PCVRP instance generation class.
    """
    @classmethod
    def get_list_of_instances(cls):
        """
        Get list of generated files.

        Args:
            n/a.

        Returns:
            None.
        """
        base_dir = path.dirname(path.dirname(path.abspath(__file__)))

        generated_root = path.join(base_dir, INSTANCES_PATH)
        try:
            generated = os.listdir(generated_root)
        except FileNotFoundError:
            return {}
        benchmark_instances = {}

        for folder in generated:
            folder_path = path.join(generated_root, folder)
            if not path.isdir(folder_path):
                continue

            folder_rel = path.join(INSTANCES_PATH, folder)
            benchmark_instances[folder] = [
                path.join(folder_rel, path.splitext(file_name)[0])
                for file_name in os.listdir(folder_path)
                if path.isfile(path.join(folder_path, file_name))
            ]
        return benchmark_instances

    def __init__(self,
                 num_agents:Optional[int]=20,
                 num_nodes:Optional[int]=100,
                 capacity:Optional[int]=50,
                 service_times:Optional[float]=0.2,
                 speed:Optional[float]=1.0,
                 profits:Optional[str]='distance',
                 instance_name:Optional[str]='validation',
                 list_of_instances:Optional[set]=None,
                 device: Optional[str] = "cpu",
                 batch_size: Optional[torch.Size] = None,
                 seed:Optional[int]=None) -> None:
        """
        Constructor. Instance generator.

        Args:
            num_agents(int): Total number of agents. Defaults to 20.
            num_nodes(int):  Total number of nodes. Defaults to 100.
            capacity(int): Total capacity for each agent. Defaults to 50.
            service_times(int): Service time in the nodes. Defaults to 0.2.
            speed(float): Vehicles' speed. Defaults to None.
            profits(str): Type of profits to use. It can be 'constant', 'uniform' or 'distance'. Defaults to 'distance'.
            instance_name(str): Instance name. Defaults to "validation".
            list_of_instances(set):  List of instances file names. Defaults to None.
            device(str, optional): Type of processing. It can be "cpu" or "gpu". Defaults to "cpu".
            batch_size(torch.Size, optional): Batch size. If not specified, defaults to 1.
            seed(int): Random number generator seed. Defaults to None.

        Returns:
            None.
        """

        # seed the generation process
        if seed is None:
            self._set_seed(self.DEFAULT_SEED)
        else:
            self._set_seed(seed)

        if num_agents is not None:
            assert num_agents>0, f"number of agents must be grater them 0!"
            self.num_agents = num_agents
        if num_nodes is not None:
            assert num_nodes>0, f"number of services must be grater them 0!"
            self.num_nodes = num_nodes
        if service_times is not None:
            self.service_times = service_times
        if capacity is not None:
            assert capacity>0, f"agent capacity must be grater them 0!"
            self.capacity = capacity
        if speed is not None:
            assert speed>0, f'Speed must be greater than 0!'
            self.speed = speed

        self.profits = profits

        self.device = device
        if batch_size is None:
            batch_size = [1]
        else:
            batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        self.batch_size = torch.Size(batch_size)

        assert instance_name in ["validation", "test"], f"instance unknown type"
        self.list_of_instances = list_of_instances
        if list_of_instances:
            self.instance_name = instance_name
            self.load_list_of_instances()


    def read_instance_data(self, instance_name:str)-> Dict:
        """
        Read instance data from file.

        Args:
            instance_name(str): instance file name.

        Returns:
            Dict: Instance data.
        """

        base_dir = path.dirname(path.dirname(path.abspath(__file__)))
        generated_file = '{path_to_generated_instances}/{instance}.pkl' \
                        .format(path_to_generated_instances=base_dir,
                                instance=instance_name)
        with open(generated_file, 'rb') as fp:
            instance = pickle.load(fp)

        self.batch_size = instance['data'].batch_size
        instance['data'] = instance['data'].to(self.device)
        return instance


    def get_instance(self, instance_name:str, num_agents:int=None) -> Dict:
        """
        Get an instance with custom number of agents.

        Args:
            instance_name(str): Instance file name.
            num_agents(int): Number of agents. Defaults to None.

        Returns:
            Dict: Instance data.
        """
        instance = self.instances_data.get(instance_name)

        if num_agents is not None:
            assert num_agents>0, f"number of agents must be grater them 0!"
            instance['num_agents'] = num_agents

        return instance

    def load_list_of_instances(self, list_of_instances:list=None):
        """
        Load every instance on list_of_instances list.

        Args:
            list_of_instances(list): List of instances file names. Defaults to None.

        Returns:
            None.
        """
        if list_of_instances:
            self.list_of_instances = list_of_instances
        self.instances_data = dict()
        for instance_name in self.list_of_instances:
            instance = self.read_instance_data(instance_name)
            self.instances_data[instance_name] = instance



    def random_generate_instance(self,
                                 batch_size: Optional[torch.Size] = None,
                                 seed:Optional[int]=None,
                                 device:Optional[str]="cpu")-> TensorDict:
        """
        Generate random instance.

        Args:
            batch_size(torch.Size, optional): Batch size. Defaults to None.
            seed(int, optional): Random number generator seed. Defaults to None.

        Returns:
            TensorDict: Instance data.
        """
        if seed is not None:
            self._set_seed(seed)


        if batch_size is not None:
            batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
            self.batch_size = torch.Size(batch_size)

        instance = TensorDict({}, batch_size=self.batch_size, device=self.device)

        self.depot_idx = 0
        self.max_time = 3

        instance['depot_idx'] = self.depot_idx * torch.ones((*self.batch_size, 1), dtype = torch.int64, device=self.device)

        coords = torch.rand(*self.batch_size, self.num_nodes, 2, dtype = torch.float, device=self.device)
        instance['coords'] = coords

        demands = torch.randint(low = 1, high=11, size = (*self.batch_size, self.num_nodes), dtype = torch.float, device=self.device)
        demands[:, self.depot_idx] = 0.0

        instance['demands'] = demands
        service_times = self.service_times * torch.ones((*self.batch_size, self.num_nodes), dtype = torch.float, device=self.device)
        service_times[:, self.depot_idx] = 0
        instance['service_time'] = service_times
        instance['speed'] = torch.full((*self.batch_size, 1), self.speed, dtype=torch.float32)


        instance['is_depot'] = torch.zeros((*self.batch_size, self.num_nodes), dtype=torch.bool, device=self.device)
        instance['is_depot'][:, self.depot_idx] = True

        instance['start_time'] = torch.zeros(*self.batch_size, dtype=torch.float, device=self.device)
        instance['end_time'] = self.max_time * torch.ones(*self.batch_size, dtype=torch.float, device=self.device)

        if self.profits == 'constant':
            self.profits = torch.ones((*self.batch_size, self.num_nodes), dtype = torch.float, device=self.device) # constant
        elif self.profits == 'uniform':
            self.profits = torch.randint(low = 1, high=100, size = (*self.batch_size, self.num_nodes), dtype = torch.float, device=self.device) / 100 # uniform
        elif self.profits == 'distance':
            depot_loc = coords.gather(1, instance['depot_idx'][:,:,None].expand(-1, -1, 2))
            depot2nodes = torch.pairwise_distance(depot_loc, coords, eps=0, keepdim = False)
            self.profits = (1+ torch.floor(99 * depot2nodes / torch.max(depot2nodes, dim=1, keepdim = True).values)) / 100 # distance
        self.profits[:, self.depot_idx] = 0
        instance['profits'] = self.profits

        instance['capacity'] = self.capacity * torch.ones((*self.batch_size, 1), dtype = torch.float, device=self.device)

        instance_info = {'name':'random_instance',
                         'num_nodes': self.num_nodes,
                         'num_agents':self.num_agents,
                         'data':instance}
        return instance_info

    def augment_generate_instance(self,
                                 batch_size: Optional[torch.Size] = None,
                                 n_augment:Optional[int] = 2,
                                 seed:Optional[int]=None,
                                 device:Optional[str]="cpu")-> TensorDict:
        """
        Generate augmentated instance.

        Args:
            batch_size(torch.Size, optional): Batch size. Defaults to None.
            n_augment(int): Data augmentation. Defaults to 2.
            seed(int, optional): Random number generator seed. Defaults to None.

        Returns:
            TensorDict: Instance data.
        """
        if seed is not None:
            self._set_seed(seed)

        if batch_size is not None:
            batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
            self.batch_size = torch.Size(batch_size)

        assert self.batch_size.numel()%n_augment == 0, f"batch_size must be divisible by n_augment"
        s_batch_size = self.batch_size.numel() // n_augment
        self.s_batch_size = torch.Size([s_batch_size])

        instance_info_s = self.random_generate_instance(batch_size = self.s_batch_size,
                                                     seed=seed,
                                                     device=device)

        self.batch_size = torch.Size(batch_size)

        instance = TensorDict({}, batch_size=self.batch_size, device=self.device)
        for key in instance_info_s['data'].keys():
            if len(instance_info_s['data'][key].shape) == 3:
                instance[key] = instance_info_s['data'][key].repeat(n_augment, 1, 1)
            elif len(instance_info_s['data'][key].shape) == 2:
                instance[key] = instance_info_s['data'][key].repeat(n_augment, 1)
            elif len(instance_info_s['data'][key].shape) == 1:
                instance[key] = instance_info_s['data'][key].repeat(n_augment)

        instance_info = {'name':'random_instance',
                         'num_nodes': self.num_nodes,
                         'num_agents':self.num_agents,
                         'data':instance}
        return instance_info


    def sample_name_from_list(self, seed:Optional[int]=None)-> str:
        """
        Sample one instance from instance list.

        Args:
            seed(int, optional): Random number generator seed. Defaults to None.

        Returns:
            str: Instance name.
        """
        if seed is not None:
            self._set_seed(seed)
        assert len(self.list_of_instances)>0, f"list_of_instances has to have at least one instance!"

        return random.choice(self.list_of_instances)

    def sample_instance(self,
                        num_agents:Optional[int]=None,
                        num_nodes:Optional[int]=None,
                        capacity:Optional[int]=50,
                        service_times:Optional[float]=0.2,
                        speed:Optional[float]=None,
                        profits:Optional[str]='constant',
                        instance_name:Optional[str]=None,
                        sample_type:Optional[str]='random',
                        batch_size: Optional[torch.Size] = None,
                        n_augment: Optional[int] = None,
                        seed:Optional[int]=None,
                        device: Optional[str] = "cpu")-> Dict:
        """
        Sample one instance from instance space.

        Args:
            num_agents(int): Total number of agents. Defaults to None.
            num_nodes(int):  Total number of nodes. Defaults to None.
            capacity(int): Total capacity for each agent. Defaults to 50.
            service_times(int): Service time in the nodes. Defaults to 0.2.
            speed(float): Vehicles' speed. Defaults to 1.0.
            profits(str): Type of profits to use. It can be 'constant', 'uniform' or 'distance'. Defaults to 'constant'.
            instance_name(str):  Instance name. Defaults to None.
            sample_type(str): Sample type. It can be "random", "augment" or "saved". Defaults to "random".
            batch_size(torch.Size, optional): Batch size. Defaults to None.
            n_augment(int, optional): Data augmentation. Defaults to None.
            seed(int): Random number generator seed. Defaults to None.

        Returns:
            Dict: Instance data.
        """
        if seed is not None:
            self._set_seed(seed)

        if self.list_of_instances is None:
            random_sample = True
        else:
            random_sample = False

        if instance_name==None and random_sample==False:
            instance_name = self.sample_name_from_list(seed=seed)
        elif instance_name==None and random_sample==True:
            instance_name = 'random_instance'
        else:
            instance_name = instance_name


        if num_agents is not None:
            self.num_agents = num_agents
        if num_nodes is not None:
            self.num_nodes = num_nodes
        if capacity is not None:
            self.capacity = capacity
        if service_times is not None:
            self.service_times = service_times
        if speed is not None:
            self.speed = speed
        if profits != "distance":
            self.profits = profits

        if batch_size is not None:
            batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
            self.batch_size = torch.Size(batch_size)

        if sample_type=='random':
            instance_info = self.random_generate_instance(batch_size = batch_size,
                                                     seed=seed,
                                                     device=device)
        elif sample_type=='augment':
            instance_info = self.augment_generate_instance(batch_size = batch_size,
                                                     n_augment = n_augment,
                                                     seed=seed,
                                                     device=device)
        elif sample_type=='saved':
            instance_info = self.get_instance(instance_name, num_agents=num_agents)

        return instance_info

if __name__ == '__main__':

    number_instances = 64
    print('starting validation set generation')

    # validation set generation
    for num_nodes, n_agent in [(101, 5), (51, 5)]:
        generator = InstanceGenerator(batch_size=32, seed=0)
        for k in range(number_instances):
            instance =  generator.sample_instance(num_agents=n_agent, num_nodes=num_nodes)
            name = f'generated_val_servs_{num_nodes-1}_agents_{n_agent}_{k}'
            instance['name'] = name
            if not os.path.exists(f'data/generated/val_servs_{num_nodes-1}_agents_{n_agent}'):
                os.makedirs(f'data/generated/val_servs_{num_nodes-1}_agents_{n_agent}')
            with open(f'data/generated/val_servs_{num_nodes-1}_agents_{n_agent}/'+name+'.pkl', 'wb') as fp:
                pickle.dump(instance, fp, protocol=pickle.HIGHEST_PROTOCOL)

    print('done')
