import torch
from tensordict import TensorDict

import os
from os import path
import pickle

from typing import Dict, Optional

from maenvs4vrp.core.env_generator_builder import InstanceBuilder
from maenvs4vrp.utils.utils import gather_by_index

GENERATED_INSTANCES_PATH = 'cvrptw/data/generated'


class InstanceGenerator(InstanceBuilder):
    """
    class for CVRPTW benchmark instances generation
    
    """
    
    def __init__(self, 
                 instance_type:str='validation', 
                 set_of_instances:set=None, 
                 device: Optional[str] = "cpu",
                 batch_size: Optional[torch.Size] = None,
                 seed:int=None) -> None:
        """    
        Constructor

        Args:       
            instance_type(str): Instance type. Can be "validation" or "test". Defaults to "validation".
            set_of_instances(bool): Set of instances file names. Defaults to None.
            device(str, optional): Device. Defaults to "cpu".
            batch_size(torch.Size, optional): ***Description. Defaults to None.
            seed(int): Random number generator seed. Defaults to None.
        """

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

        self.max_num_agents = 20
        self.max_num_nodes = 100

        assert instance_type in ["validation", "test"], f"instance unknown type"
        self.set_of_instances = set_of_instances
        if set_of_instances:
            self.instance_type = instance_type
            self.load_set_of_instances()
            

    def get_time_windows(self, 
                         instance:TensorDict=None,
                         horizon = 480, 
                         tw_ratio = 0.75,
                         cust_tw_range = (30,91),
                         seed:int=None)-> torch.tensor:
        """
        Get time windows

        Args:
            instance(TensorDict): ***Description. Defaults to None.
            horizon(float): Horizon. Defaults to 480.
            tw_ratio(float): Tw ratio. Defaults to 0.75.
            cust_tw_range: Tw range cost. Defaults to 30.91.
            seed(int): Random number generator seed. Defaults to None.

        Returns: 
            np.array: nodes time windows;
        """

        if seed is not None:
            self._set_seed(seed)

        time_windows = torch.zeros((*self.batch_size, self.max_num_nodes, 2), device=self.device)
        
        size = (*self.batch_size, self.max_num_nodes, 1)

        # Sample TW subset ~ B(tw_ratio)
        if isinstance(tw_ratio, float):
            has_tw = torch.empty(size, device=self.device).bernoulli_(tw_ratio)
        elif len(tw_ratio) == 1:
            has_tw = torch.empty(size, device=self.device).bernoulli_(tw_ratio[0])
        else: # tuple of float
            ratio = torch.tensor(tw_ratio)[torch.randint(0, len(tw_ratio), (*self.batch_size,), device=self.device, dtype = torch.int64)]
            has_tw = ratio[:,None,None].expand(*size).bernoulli()
        
        has_tw[:, self.depot_idx, :] = 0
        # Sample TW width        tw_j = H if not in TW subset
        #                        tw_j ~ U(30,90) if in TW subset
        tws = (1 - has_tw) * torch.full(size, horizon, device=self.device) \
                + has_tw * torch.randint(*cust_tw_range, size, dtype = torch.float, device=self.device)

        tts = torch.pairwise_distance(instance['coords'][:, self.depot_idx, None], instance['coords'][:], eps=0, keepdim = True) 
        
        # Sample ready time       e_j = 0 if not in TW subset
        #                         e_j ~ U(a_j, H - max(tt_0j + s_j, tw_j))
        rdys = has_tw * (torch.rand(size, device=self.device) * (horizon - torch.max(tts + instance['service_time'].unsqueeze(-1), tws)))
        rdys.floor_()

        time_windows[:, :, 0] = rdys.squeeze(-1)
        time_windows[:, :, 1] = (rdys + tws).squeeze(-1)
        time_windows[:, self.depot_idx, 0] = 0
        time_windows[:, self.depot_idx, 1] = horizon
        return time_windows


    def random_generate_instance(self, num_agents:int=20, 
                                 num_nodes:int=100, 
                                 capacity:int=200, 
                                 service_times:int=10, 
                                 batch_size: Optional[torch.Size] = None,
                                 seed:int=None)-> TensorDict:
        """
        CVRPTW random instances generated following:

        @inproceedings{li2021learning,
                        title={Learning to delegate for large-scale vehicle routing},
                        author={Sirui Li and Zhongxia Yan and Cathy Wu},
                        booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
                        year={2021}
                        }
        see: https://github.com/mit-wu-lab/learning-to-delegate/blob/main/generate_initial.py

        Args:
            num_nodes(int, optional):  Total number of nodes. Defaults to 100.
            num_agents(int, optional): Total number of agents. Defaults to 20.
            capacity(int, optional): Total capacity for each agent. Defaults to 200.
            service_times(int, optional): Total time of service. Defaults to 10.
            batch_size(torch.Size, optional): ***Description. Defaults to None.
            seed(int, optional): random number generator seed. Defaults to None.

        Returns:
            Dict: Instance data
        """
        if seed is not None:
            self._set_seed(seed)

        if num_agents is not None:
            assert num_agents>0, f"number of agents must be grater them 0!"
            self.max_num_agents = num_agents
        if num_nodes is not None:
            assert num_nodes>0, f"number of services must be grater them 0!"
            self.max_num_nodes = num_nodes
        if service_times is not None:
            self.service_times = service_times
        if capacity is not None:
            assert capacity>0, f"agent capacity must be grater them 0!"
            self.capacity = capacity

        if batch_size is not None:
            batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
            self.batch_size = torch.Size(batch_size)

        instance = TensorDict({}, batch_size=self.batch_size, device=self.device)
        
        self.depot_idx = 0
        instance['depot_idx'] = self.depot_idx * torch.ones((*self.batch_size, 1), dtype = torch.int64, device=self.device)
        
        # Sample locs  x_j, y_j ~ U(0, 100)
        coords = torch.randint(low = 0, high=101, size = (*self.batch_size, self.max_num_nodes, 2), dtype = torch.float, device=self.device)        
        instance['coords'] = coords

        # Sample dems  q_j ~ U(5,  40)
        demands = torch.randint(low = 5, high=41, size = (*self.batch_size, num_nodes), dtype = torch.float, device=self.device)
        demands[:, self.depot_idx] = 0.0
        instance['demands'] = demands

        # Sample serv. time  s_j ~ U(10, 30)
        service_times = torch.randint(low = 10, high=31, size = (*self.batch_size, num_nodes), dtype = torch.float, device=self.device)
        service_times[:, self.depot_idx] = 0
        instance['service_time'] = service_times

        time_windows = self.get_time_windows(instance=instance, seed=seed)

        instance['tw_low'] =  time_windows[:, :, 0].clone()
        instance['tw_high'] = time_windows[:, :, 1].clone()

        instance['is_depot'] = torch.zeros((*self.batch_size, num_nodes), dtype=torch.bool, device=self.device)
        instance['is_depot'][self.depot_idx] = True

        instance['start_time'] = time_windows[:, :, 0].gather(1, torch.zeros((*self.batch_size, 1), 
                                                                          dtype=torch.int64, device=self.device)).squeeze(-1)
        instance['end_time'] = time_windows[:, :, 1].gather(1, torch.zeros((*self.batch_size, 1), 
                                                                        dtype=torch.int64, device=self.device)).squeeze(-1)

        instance['capacity'] = self.capacity * torch.ones((*self.batch_size, 1), dtype = torch.float, device=self.device)

        instance_info = {'name':'random_instance',
                         'num_nodes': self.max_num_nodes,
                         'num_agents':self.max_num_agents,
                         'data':instance}
        return instance_info



    def sample_instance(self, 
                        num_agents=20, 
                        num_nodes=100, 
                        capacity=200, 
                        service_times=10, 
                        instance_name:str=None, 
                        random_sample:bool=True,
                        batch_size: Optional[torch.Size] = None,
                        seed:int=None)-> Dict:
        """
        Samples one instance from instance space

        Args:
            num_agents(int): Total number of agents. Defaults to 20.
            num_nodes(int):  Total number of nodes. Defaults to 100.
            capacity(int): Total capacity for each agent. Defaults to 200.
            service_times(int): Total time of service. Defaults to 10.            
            instance_name(str):  instance name. Defaults to None.
            random_sample(bool):  True to sample instance and False to use original instance data. Defaults to None.
            batch_size(torch.Size, optional): Batch size. Defaults to None.
            seed (int): random number generator seed. Defaults to None.

        Returns:
            Dict: Instance data.
        """
        if seed is not None:
            self._set_seed(seed)

        if self.set_of_instances is None:
            random_sample = True
        else:
            random_sample = False

        if instance_name is None:
            random_sample = True
        else:
            random_sample = False  

        if num_agents is None:
            num_agents = 20
        if num_nodes is None:
            num_nodes = 100
        if capacity is None:
            capacity = 50
        if service_times is None:
            service_times = 0.2

        if instance_name==None and random_sample==False:
            self.instance_name = self.sample_one_instance(seed=seed)
        elif instance_name==None and random_sample==True:
            self.instance_name = 'random_instance'
        else:
            self.instance_name = instance_name

        if batch_size is not None:
            batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
            self.batch_size = torch.Size(batch_size)

        if random_sample:
            instance_info = self.random_generate_instance(num_agents=num_agents, 
                                                     num_nodes=num_nodes, 
                                                     capacity=capacity, 
                                                     service_times=service_times,
                                                     batch_size = batch_size,
                                                     seed=seed)
        else:
            instance = self.get_instance(self.instance_name, num_agents=num_agents, preloaded=True)

        return instance_info

if __name__ == '__main__':

    number_instances = 128
    print('starting valid/test sets generation')

    if not os.path.exists('data/generated/test'):
        os.makedirs('data/generated/test')
    if not os.path.exists('data/generated/validation'):
        os.makedirs('data/generated/validation')

    # valid/test sets generation
    for n_serv, n_agent in [(25, 10), (50, 20), (100, 30), (250, 40), (500, 50), (1000, 100)]:
        generator = InstanceGenerator(seed=0)

        for k in range(number_instances):
            instance =  generator.sample_instance(num_agents=n_agent, num_services=n_serv)
            del instance['duration_matrix']
            name = f'generated_val_servs_{n_serv}_agents_{n_agent}_{k}'
            instance['name'] = name
            with open('data/generated/validation/'+name+'.pkl', 'wb') as fp:
                pickle.dump(instance, fp, protocol=pickle.HIGHEST_PROTOCOL)

        for k in range(number_instances):
            instance =  generator.sample_instance(num_agents=n_agent, num_services=n_serv)
            del instance['duration_matrix'] 
            name = f'generated_test_servs_{n_serv}_agents_{n_agent}_{k}'
            instance['name'] = name
            with open('data/generated/test/'+name+'.pkl', 'wb') as fp:
                pickle.dump(instance, fp, protocol=pickle.HIGHEST_PROTOCOL)

    print('done')
