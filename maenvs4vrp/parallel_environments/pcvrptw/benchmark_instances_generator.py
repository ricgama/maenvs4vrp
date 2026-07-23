import torch
from tensordict import TensorDict

import os
from os import path

from typing import Dict, Optional
from maenvs4vrp.core.env_generator_builder import InstanceBuilder


import warnings
import shutil
import random
from huggingface_hub import HfApi, snapshot_download

HF_REPO_ID = "maenvs4vrp/environments"
INSTANCES_PATH = 'cvrpstw/data/benchmark'
DATA_PATH = './pcvrptw/data/benchmark'


class BenchmarkInstanceGenerator(InstanceBuilder):
    """
    PCVRPTW benchmark instance generation class.
    """
    @classmethod
    def get_list_of_instances(cls):
        """
        Get list of possible instances from benchmark files.

        Args:
            n/a.

        Returns:
            None.
        """
        base_dir = path.dirname(path.dirname(path.abspath(__file__)))

        return {'Solomon': [s.split('.')[0] for s in os.listdir(path.join(base_dir, DATA_PATH, 'Solomon'))],
                'Homberger':[s.split('.')[0] for s in os.listdir(path.join(base_dir, DATA_PATH, 'Homberger'))]}

    def __init__(self,
                 num_agents:Optional[int]=None,
                 num_nodes:Optional[int]=None,
                 speed:Optional[float]=1.0,
                 instance_name:Optional[str]='Solomon',
                 list_of_instances:Optional[list]=None,
                 device: Optional[str] = "cpu",
                 batch_size: Optional[torch.Size] = None,
                 seed:Optional[int]=None) -> None:
        """
        Constructor. Create an instance space of one or several sets of data.

        Args:

            instance_name(str): Instance name. Can be "Solomon" or "Homberger". Defaults to "Solomon".
            list_of_instances(list): List of instances file names. Defaults to None.
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

        self.num_agents = num_agents
        self.num_nodes = num_nodes
        self.speed = speed

        self.device = device
        if batch_size is None:
            batch_size = [1]
        else:
            batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        self.batch_size = torch.Size(batch_size)

        assert instance_name in ["Solomon", "Homberger"], f"instance unknown type"
        self.instance_name = instance_name

        dataset_available = self._ensure_dataset_exists()
        assert dataset_available, f"dataset is not available for instance type '{self.instance_name}'"

        if list_of_instances is not None:
            self.list_of_instances = list_of_instances
        else:
            self.list_of_instances = self.get_list_of_instances().get(instance_name, [])

        self.load_list_of_instances()

    def _ensure_dataset_exists(self) -> bool:
        """
        Ensure dataset (or a dataset subfolder) exists locally.
        """

        base_dir = path.dirname(path.dirname(path.abspath(__file__)))
        target_dir = path.join(base_dir, DATA_PATH)
        local_instance_dir = path.join(target_dir, self.instance_name) if self.instance_name is not None else None

        # Dataset already present
        if os.path.isdir(target_dir) and (local_instance_dir is None or os.path.isdir(local_instance_dir)):
            return True

        api = HfApi()
        if not api.repo_exists(repo_id=HF_REPO_ID, repo_type="dataset"):
            warnings.warn(
                f"Dataset '{HF_REPO_ID}' is not available on Hugging Face. "
                "Falling back to random instance generation.",
                RuntimeWarning,
            )
            return False

        if self.instance_name is not None:
            repo_files = api.list_repo_files(repo_id=HF_REPO_ID, repo_type="dataset")

            candidate_roots = [
                INSTANCES_PATH,
                f"environments/{INSTANCES_PATH}"
            ]
            repo_instance_dir = None
            for root in candidate_roots:
                candidate = f"{root}/{self.instance_name}"
                if any(repo_file.startswith(f"{candidate}/") for repo_file in repo_files):
                    repo_instance_dir = candidate
                    break

            if repo_instance_dir is None:
                warnings.warn(
                    f"Instance subfolder '{self.instance_name}' was not found in '{HF_REPO_ID}' under "
                    f"'{INSTANCES_PATH}'"
                    "Falling back to random instance generation.",
                    RuntimeWarning,
                )
                return False

            print(f"Instance subfolder '{self.instance_name}' not found locally. Downloading from HuggingFace...")

            snapshot_path = snapshot_download(
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                allow_patterns=[f"{repo_instance_dir}/**"],
            )

            remote_instance_dir = path.join(snapshot_path, *repo_instance_dir.split("/"))
            if not os.path.isdir(remote_instance_dir):
                raise RuntimeError(
                    f"Downloaded snapshot does not contain '{repo_instance_dir}'."
                )

            os.makedirs(target_dir, exist_ok=True)
            assert local_instance_dir is not None
            shutil.copytree(remote_instance_dir, local_instance_dir, dirs_exist_ok=True)
            print(f"Instance subfolder installed at: {local_instance_dir}")
            return True

        print("Dataset not found. Downloading snapshot from HuggingFace...")

        # Download complete repo snapshot
        snapshot_path = snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset"
            #local_dir_use_symlinks=False
        )

        print("Snapshot downloaded at:", snapshot_path)

        # Find the folder named "data" inside the snapshot
        dataset_root = None
        for root, dirs, _ in os.walk(snapshot_path):
            if "data" in dirs:
                dataset_root = path.join(root, "data")
                break

        if dataset_root is None:
            raise RuntimeError("Could not find 'data/' folder inside the snapshot.")

        print("Found dataset folder:", dataset_root)

        # Copy entire data folder preserving structure
        shutil.copytree(dataset_root, target_dir, dirs_exist_ok=True)

        print(f"Dataset installed at: {target_dir}")
        return True



    def read_instance_data(self, instance_name:str)-> Dict:
        """
        Read instance data from file.

        Args:
            instance_name(str): Instance file name.

        Returns:
            Dict: Instance data.
        """

        base_dir = path.dirname(path.dirname(path.abspath(__file__)))

        path_to_file = path.join(base_dir, DATA_PATH, self.instance_name)

        benchmark_file = '{path_to_benchmark_instances}/{instance}.txt' \
                        .format(path_to_benchmark_instances=path_to_file,
                                instance=instance_name)

        dfile = open(benchmark_file)

        data = [[x for x in line.split()] for line in dfile]
        dfile.close()
        instance = self.parse_instance_data(data)
        return instance

    def parse_instance_data(self, instance_data: list) -> Dict:
        """
        Parse instance data list into a dictionary.

        Args:
            instance_data(list): Instance data.

        Returns:
            Dict: Parsed instance data.
        """
        instance = dict()
        instance['name'] = instance_data[0][0]

        coords = []
        demands = []
        time_windows = []
        service_time = []

        for data in instance_data[9:]:
            coords.append([float(data[1]), float(data[2])])
            demands.append(float(data[3]))
            time_windows.append([float(data[4]), float(data[5])])
            service_time.append(float(data[6]))

        instance['num_agents'] = int(instance_data[4][0])
        instance['num_nodes'] = len(coords) # -1 to account for depot

        data = TensorDict({}, batch_size=self.batch_size, device=self.device)

        capacity = float(instance_data[4][1])

        depot_idx = 0
        data['depot_idx'] = depot_idx * torch.ones((*self.batch_size, 1), dtype = torch.int64, device=self.device)
        data['coords'] = torch.tensor(coords, dtype = torch.float, device=self.device).unsqueeze(0)

        data['demands'] = torch.tensor(demands, dtype = torch.float, device=self.device).unsqueeze(0)
        profits = torch.clip(torch.zeros_like(data['demands'], device=self.device).uniform_(0.75, 2.25)*data['demands'], min=1)
        profits[:, depot_idx] = 0.0
        data['profits'] = torch.round(profits)
        time_windows = torch.tensor(time_windows, dtype = torch.float, device=self.device).unsqueeze(0)
        data['tw_low'] =  time_windows[:, :, 0].clone()
        data['tw_high'] = time_windows[:, :, 1].clone()

        data['service_time'] = torch.tensor(service_time, dtype = torch.float, device=self.device).unsqueeze(0)
        data['start_time'] = time_windows[:, :, 0].gather(1, torch.zeros((*self.batch_size, 1),
                                                                          dtype=torch.int64, device=self.device)).squeeze(-1)
        data['end_time'] = time_windows[:, :, 1].gather(1, torch.zeros((*self.batch_size, 1),
                                                                        dtype=torch.int64, device=self.device)).squeeze(-1)
        data['speed'] = torch.ones((*self.batch_size, 1), dtype=torch.float, device=self.device)

        data['is_depot'] = torch.zeros((*self.batch_size, instance['num_nodes']), dtype=torch.bool, device=self.device)
        data['is_depot'][:, depot_idx] = True
        data['capacity'] = capacity * torch.ones((*self.batch_size, 1), dtype = torch.float, device=self.device)

        instance['data'] = data

        if self.instance_name in ['Solomon', 'Homberger']:
            instance['n_digits'] = 10.0
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
            num_agents = min(instance['num_agents'], num_agents)
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


    def sample_first_n_services(self,
                                device:Optional[str]="cpu")-> Dict:
        """
        Sample first n nodes.

        Args:
            device(str, optional): Type of processing. It can be "cpu" or "gpu". Defaults to "cpu".
        Returns:
            Dict: New instance of the first n nodes.
        """

        new_instance = dict()
        instance = self.get_instance(self.instance_name, self.num_agents)

        new_instance['num_agents'] = instance['num_agents']

        if self.num_nodes is not None:
            self.num_nodes = min(self.num_nodes, instance['num_nodes'])
            new_instance['num_nodes'] = self.num_nodes
        else:
            new_instance['num_nodes'] = instance['num_nodes']

        new_instance['name'] = instance['name'] + '_samp'
        new_instance['n_digits'] = instance['n_digits']
        data = instance['data']

        idxs = torch.arange(0, instance['num_nodes'], device=self.device)
        index = idxs[:new_instance['num_nodes']]

        new_data = TensorDict({}, batch_size=self.batch_size, device=self.device)

        new_data['capacity'] = data['capacity']
        new_data['depot_idx'] = data['depot_idx']
        new_data['coords'] = data['coords'][:, index]
        new_data['demands'] = data['demands'][:,index]
        new_data['profits'] = data['profits'][:,index]
        new_data['tw_low'] = data['tw_low'][:,index]
        new_data['tw_high'] = data['tw_high'][:,index]
        new_data['service_time'] = data['service_time'][:,index]
        new_data['start_time'] = data['start_time']
        new_data['end_time'] = data['end_time']
        new_data['is_depot'] = data['is_depot'][:, index]
        new_data['speed'] = data['speed']

        new_instance['data'] = new_data

        return new_instance

    def random_sample_instance(self,
                               seed:int=None,
                               device:Optional[str]="cpu")-> Dict:
        """
        Sample one instance from instance space, randomly adjusting the nodes.

        Args:
            instance_name(Optional[str]): Instance file name. Defaults to None.
            num_agents(Optional[int]):  Total number of agents. Defaults to None.
            num_nodes(Optional[int]):  Total number of nodes. Defaults to None.
            speed(Optional[float]): Vehicles' speed. Defaults to None.
            seed(Optional[int]): Random number generator seed. Defaults to None.

        Returns:
            Dict: Instance data.
        """
        if seed is not None:
            self._set_seed(seed)

        new_instance = dict()
        instance = self.get_instance(self.instance_name, self.num_agents)

        new_instance['num_agents'] = instance['num_agents']

        if self.num_nodes is not None:
            self.num_nodes = min(self.num_nodes, instance['num_nodes'])
            new_instance['num_nodes'] = self.num_nodes
        else:
            new_instance['num_nodes'] = instance['num_nodes']

        new_instance['name'] = instance['name'] + '_samp'

        new_instance['n_digits'] = instance['n_digits']

        data = instance['data']

        idxs = torch.arange(0, instance['num_nodes'], device=self.device).unsqueeze(0)
        index = torch.cat([idxs[data['is_depot']], idxs[~data['is_depot']][torch.randperm(instance['num_nodes']-1)]], dim=0)[:new_instance['num_nodes']]

        new_data = TensorDict({}, batch_size=self.batch_size, device=self.device)

        new_data['capacity'] = data['capacity']
        new_data['depot_idx'] = data['depot_idx']
        new_data['coords'] = data['coords'][:, index]
        new_data['demands'] = data['demands'][:,index]
        new_data['profits'] = data['profits'][:,index]
        new_data['tw_low'] = data['tw_low'][:,index]
        new_data['tw_high'] = data['tw_high'][:,index]
        new_data['service_time'] = data['service_time'][:,index]
        new_data['start_time'] = data['start_time']
        new_data['end_time'] = data['end_time']
        new_data['is_depot'] = data['is_depot'][:, index]
        new_data['speed'] = data['speed']

        new_instance['data'] = new_data
        return new_instance

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
                        capacity:Optional[int]=None,
                        service_times:Optional[float]=None,
                        speed:Optional[float]=1.0,
                        profits:Optional[str]=None,
                        instance_name:Optional[str]=None,
                        sample_type:Optional[str]='random',
                        batch_size: Optional[torch.Size] = None,
                        n_augment: Optional[int] = None,
                        seed:Optional[int]=None,
                        device:Optional[str]="cpu")-> Dict:
        """
        Sample one instance from instance space.

        Args:
            num_agents(int): Total number of agents. Defaults to None.
            num_nodes(int): Total number of nodes. Defaults to None.
            capacity(int): Capacity of the agents. Defaults to None.
            service_times(float): Service time in the nodes. Defaults to None.
            speed(float): Vehicles' speed. Defaults to None.
            instance_name(str): Instance name. Defaults to None.
            sample_type(str): Sample type. It can be "random" or something else for "first n". Defaults to "random".
            batch_size(torch.Size or None): Batch size. Defaults to None.
            n_augment(int, optional): Data augmentation. Defaults to None.
            seed(Optional[int]): Random number generator seed. Defaults to None.

        Returns:
            Dict: Instance data.
        """
        if seed is not None:
            self._set_seed(seed)

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

        if instance_name==None:
            instance_name = self.sample_name_from_list(seed=seed)
        else:
            instance_name = instance_name

        if sample_type=='random':
            instance = self.random_sample_instance(seed=seed,
                                                   device=device)
        else:
            # sample first n
            instance = self.sample_first_n_services(device=device)

        return instance


if __name__ == '__main__':

    generator = BenchmarkInstanceGenerator(instance_name='Solomon', list_of_instances={'C101'})
    generator.sample_instance(num_agents=3, num_nodes=8, seed=1)
