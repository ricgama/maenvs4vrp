import random

import torch
from tensordict import TensorDict

import os
from os import path
import pickle

from typing import Dict, Optional
from maenvs4vrp.core.env_generator_builder import InstanceBuilder

import warnings

from huggingface_hub import HfApi, snapshot_download
import shutil

HF_REPO_ID = "maenvs4vrp/environments"
INSTANCES_PATH = 'cvrpstw/data/generated'
DATA_PATH = './cvrpstw/data/generated'

class InstanceGenerator(InstanceBuilder):
    """
    CVRPSTW instance generation class.
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
                 instance_name:Optional[str]="train",
                 list_of_instances:Optional[list]=None,
                 device: Optional[str] = "cpu",
                 batch_size: Optional[torch.Size] = None,
                 seed:Optional[int]=None) -> None:
        """
        Constructor. Instance generator.

        Args:
            instance_name(str):  Instance type. Can be "train", "validation", or "test". Defaults to "train".
            list_of_instances(list):  List of instances file names. Defaults to None.
            device(str, optional): Type of processing. It can be "cpu" or "gpu". Defaults to "cpu".
            batch_size(torch.Size, optional): Batch size. If not specified, defaults to 1.
            seed(Optional[int]): Random number generator seed. Defaults to None.

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

        self.device = device
        if batch_size is None:
            batch_size = [1]
        else:
            batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        self.batch_size = torch.Size(batch_size)


        assert instance_name in ["train", "validation", "test"], f"instance unknown type"
        self.list_of_instances = list_of_instances
        self.instance_name = instance_name

        if self.instance_name is not None:
            dataset_available = self._ensure_dataset_exists()
            if not dataset_available:
                self.instance_name = None
                self.list_of_instances = None
            elif list_of_instances is not None:
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



    def read_instance_data(self, instance_name:Optional[str]=None)-> Dict:
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


    def get_instance(self, instance_name:Optional[str]=None, num_agents:Optional[int]=None) -> Dict:
        """
        Get an instance with custom number of agents.

        Args:
            instance_name(Optional[str]): Instance file name.
            num_agents(Optional[int]): Number of agents. Defaults to None.

        Returns:
            Dict: Instance data.
        """
        instance = self.instances_data.get(instance_name)

        if num_agents is not None:
            assert num_agents>0, f"number of agents must be grater them 0!"
            instance['num_agents'] = num_agents

        return instance

    def load_list_of_instances(self, list_of_instances:Optional[list]=None):
        """
        Load every instance on list_of_instances list.

        Args:
            list_of_instances(List[str]): List of instances file names. Defaults to None.

        Returns:
            None.
        """
        if list_of_instances:
            self.list_of_instances = list_of_instances
        self.instances_data = dict()
        for instance_name in self.list_of_instances:
            instance = self.read_instance_data(instance_name)
            self.instances_data[instance_name] = instance


    def get_time_windows(self,
                         instance:TensorDict=None,
                         batch_size:torch.Size=None,
                         seed:int=None)-> torch.tensor:
        """
        Get time windows to reach the nodes.

        Args:
            instance(TensorDict): Data instance. Defaults to None.
            batch_size(torch.Size): Batch size. Defaults to None.
            seed(int): Random number generator seed. Defaults to None.

        Returns:
            torch.Tensor: Nodes time windows.
        """

        if seed is not None:
            self._set_seed(seed)

        time_windows = torch.zeros((*batch_size, self.num_nodes, 2), device=self.device)

        depot_coord = instance['coords'].gather(1, instance['depot_idx'][:, :, None].expand(-1, -1, 2))
        dist_depot = torch.pairwise_distance(depot_coord, instance['coords'], keepdim = True)

        depot_start, depot_end = 0, 3

        inf = depot_start + dist_depot
        sup = depot_end - dist_depot - self.service_times

        time_centers = inf.squeeze(-1) + torch.rand(*batch_size, self.num_nodes, device=self.device) * (sup-inf).squeeze(-1)
        time_half_width = torch.empty((*batch_size, self.num_nodes), device=self.device).uniform_(self.service_times / 2 , depot_end / 3)
        time_windows[:, :, 0] = torch.clip(time_centers - time_half_width, depot_start, depot_end)
        time_windows[:, :, 1] = torch.clip(time_centers + time_half_width, depot_start, depot_end)
        time_windows[:, self.depot_idx, 0] = depot_start
        time_windows[:, self.depot_idx, 1] = depot_end

        return time_windows


    def random_generate_instance(self,
                                 batch_size: Optional[torch.Size] = None,
                                 seed:Optional[int]=None,
                                 device:Optional[str]="cpu")-> TensorDict:
        """
        Generate random instance.

        Args:

            batch_size(torch.Size, optional): Batch size. Defaults to None.
            seed(Optional[int], optional): Random number generator seed. Defaults to None.
            device(Optional[str], optional): Device to use. Defaults to "cpu".

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

        time_windows = self.get_time_windows(instance, self.batch_size, seed)

        instance['tw_low'] =  time_windows[:, :, 0].clone()
        instance['tw_high'] = time_windows[:, :, 1].clone()

        instance['is_depot'] = torch.zeros((*self.batch_size, self.num_nodes), dtype=torch.bool, device=self.device)
        instance['is_depot'][:, self.depot_idx] = True

        instance['start_time'] = time_windows[:, :, 0].gather(1, torch.zeros((*self.batch_size, 1),
                                                                          dtype=torch.int64, device=self.device)).squeeze(-1)
        instance['end_time'] = time_windows[:, :, 1].gather(1, torch.zeros((*self.batch_size, 1),
                                                                        dtype=torch.int64, device=self.device)).squeeze(-1)
        instance['capacity'] = self.capacity * torch.ones((*self.batch_size, 1), dtype = torch.float, device=self.device)

        instance_info = {'name':'random_instance',
                         'num_nodes': self.num_nodes,
                         'num_agents':self.num_agents,
                         'early_penalty':1,
                         'late_penalty' : 1,
                         'Pmax' : 0.1, # fraction of max time
                         'Wmax' : 0.1, # fraction of max time
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
                         'early_penalty':1,
                         'late_penalty' : 1,
                         'Pmax' : 0.1, # fraction of max time
                         'Wmax' : 0.1, # fraction of max time
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
                        num_agents: Optional[int] = None,
                        num_nodes: Optional[int] = None,
                        capacity: Optional[int] = 50,
                        service_times: Optional[float] = 0.2,
                        speed: Optional[float] = 1.0,
                        instance_name: Optional[str] = None,
                        sample_type:str='random',
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
            instance_name(str):  Instance name. Defaults to None.
            sample_type(str): Sample type. It can be "random", "augment" or "saved". Defaults to "random".
            batch_size(torch.Size, optional): Batch size. Defaults to None.
            n_augment(int, optional): Data augmentation. Defaults to None.
            seed(Optional[int]): Random number generator seed. Defaults to None.

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

        if batch_size is not None:
            batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
            self.batch_size = torch.Size(batch_size)

        if sample_type=='random':
            instance_info = self.random_generate_instance(batch_size = self.batch_size,
                                                     seed=seed,
                                                     device=device)
        elif sample_type=='augment':
            instance_info = self.augment_generate_instance(batch_size = self.batch_size,
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
    for num_nodes, n_agent in [(101, 25), (51, 25)]:
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
