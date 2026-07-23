from numpy.random import random
import torch
from tensordict import TensorDict

import os
from os import path
import pickle

from typing import Dict, Optional
from maenvs4vrp.core.env_generator_builder import InstanceBuilder
from maenvs4vrp.utils.augment_utils import random_coord_unit_square_augment, augment_coord_by_8_fold

import warnings
import random

from huggingface_hub import HfApi, snapshot_download
import shutil

HF_REPO_ID = "maenvs4vrp/environments"
INSTANCES_PATH = 'cvrp/data/generated'
DATA_PATH = './cvrp/data/generated'


class InstanceGenerator(InstanceBuilder):
    """
    CVRP instance generation class.
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
                 num_nodes:Optional[int]=21,
                 capacity:Optional[float]=1.0,
                 speed: Optional[float] = 1.0,
                 instance_name:Optional[str]=None,
                 list_of_instances:Optional[set]=None,
                 device: Optional[str] = "cpu",
                 batch_size: Optional[torch.Size] = None,
                 seed:Optional[int]=None) -> None:
        """
        Constructor. Instance generator.

        Args:
            num_agents(Optional[int]): Total number of agents. Defaults to 20.
            num_nodes(Optional[int]): Total number of nodes. Defaults to 21.
            capacity(Optional[float]): Total capacity for each agent. Defaults to 1.0.
            speed(Optional[float]): Vehicles' speed. Defaults to 1.0.
            instance_type(Optional[str]): Instance type. Can be "train", "validation", or "test". Defaults to "train".
            list_of_instances(Optional[set]): List of instances file names. Defaults to None.
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

        self.device = device
        if batch_size is None:
            batch_size = [1]
        else:
            batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        self.batch_size = torch.Size(batch_size)

        self.num_agents = num_agents
        self.num_nodes = num_nodes

        self.capacity = capacity
        self.speed = speed

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


    def read_instance_data(self, instance_name:str)-> Dict:
        """
        Read instance data from file.

        Args:
            instance_name(str): instance file name.

        Returns:
            Dict: Instance data.
        """

        base_dir = path.dirname(path.dirname(path.abspath(__file__)))
        new_instaces_path = 'cvrp/data/generated/val_servs_50_agents_10'
        generated_file = f"{base_dir}/{new_instaces_path}/{instance_name}.pkl"

        # generated_file = '{path_to_generated_instances}/{instance}.pkl' \
        #                 .format(path_to_generated_instances=base_dir,
        #                         instance=instance_name)
        print(generated_file)
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

    def load_list_of_instances(self, list_of_instances:Optional[list]=None):
        """
        Load every instance on list_of_instances list.

        Args:
            list_of_instances(list): List of instances file names. Defaults to None.

        Returns:
            None.
        """
        if list_of_instances is not None:
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

        if device is not None:
            self.device = device

        instance = TensorDict({}, batch_size=self.batch_size, device=self.device)

        self.depot_idx = 0
        instance['depot_idx'] = self.depot_idx * torch.ones((*self.batch_size, 1), dtype = torch.int64, device=self.device)

        if self.num_nodes-1 == 20:
            demand_scaler = 30
        elif self.num_nodes-1 == 50:
            demand_scaler = 40
        elif self.num_nodes-1 == 100:
            demand_scaler = 50
        else:
            demand_scaler = 0.25 * self.num_nodes + 25
            #raise NotImplementedError

        coords = torch.rand(*self.batch_size, self.num_nodes, 2, dtype = torch.float, device=self.device)
        instance['coords'] = coords

        demands = torch.randint(1, 10, size=(*self.batch_size, self.num_nodes), dtype = torch.float, device=self.device) / float(demand_scaler)
        demands[:, self.depot_idx] = 0.0

        instance['demands'] = demands

        instance['is_depot'] = torch.zeros((*self.batch_size, self.num_nodes), dtype=torch.bool, device=self.device)
        instance['is_depot'][:, self.depot_idx] = True

        instance['capacity'] = self.capacity * torch.ones((*self.batch_size, 1), dtype = torch.float, device=self.device)

        instance['speed'] = torch.full((*self.batch_size, 1), self.speed, dtype=torch.float32)

        instance_info = {'name':'random_instance',
                         'num_nodes': self.num_nodes,
                         'num_agents':self.num_agents,
                         'data':instance}
        return instance_info

    def augment_generate_instance(self,
                                 batch_size: Optional[torch.Size] = None,
                                 n_augment:Optional[int] = 2,
                                 transform:Optional[bool] = False,
                                 seed:Optional[int]=None,
                                 device:Optional[str]="cpu")-> TensorDict:
        """
        Generate augmentated instance.

        Args:
            batch_size(torch.Size, optional): Batch size. Defaults to None.
            n_augment(int, optional): Data augmentation. Defaults to 2.
            transform(bool, optional): Whether to transform the instance. Defaults to False.
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

        if transform:
            base_coords = instance_info_s['data']['coords']  # [s_batch, num_nodes, 2]

            #aug_coords_list = [base_coords] + [
            #    random_coord_unit_square_augment(base_coords) for _ in range(n_augment - 1)
            #]
            # final shape: [batch_size, num_nodes, 2]
            #aug_coords = torch.cat(aug_coords_list, dim=0)

            aug_coords = augment_coord_by_8_fold(base_coords)
            instance['coords'] = aug_coords

        for key in instance_info_s['data'].keys():
            if key == 'coords' and transform:
                continue
            t = instance_info_s['data'][key]
            reps = (n_augment,) + (1,) * (len(t.shape) - 1)
            instance[key] = t.repeat(reps)

        instance_info = {'name':'augment_instance',
                         'num_nodes': self.num_nodes,
                         'num_agents':self. num_agents,
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
                        capacity: Optional[float] = None,
                        speed: Optional[float] = None,
                        instance_name: Optional[str] = None,
                        sample_type: Optional[str] = 'random',
                        batch_size: Optional[torch.Size] = None,
                        n_augment: Optional[int] = None,
                        transform: Optional[bool] = False,
                        seed:Optional[int]=None,
                        device:Optional[str] = "cpu")-> Dict:
        """
        Sample one instance from instance space.

        Args:
            num_agents(int, optional): Total number of agents. Defaults to None.
            num_nodes(int, optional):  Total number of nodes. Defaults to None.
            capacity(float, optional): Total capacity for each agent. Defaults to None.
            speed(float, optional): Vehicles' speed. Defaults to None.
            instance_name(str, optional):  Instance name. Defaults to None.
            sample_type(str, optional): Sample type. It can be "random", "augment" or "saved". Defaults to "random".
            batch_size(torch.Size, optional): Batch size. Defaults to None.
            n_augment(int, optional): Data augmentation. Defaults to None.
            seed(int, optional): Random number generator seed. Defaults to None.
            transform(bool, optional): Whether to apply coordinate transformation. Defaults to False.

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
                                                     transform = transform,
                                                     seed=seed,
                                                     device=device)
        elif sample_type=='saved':
            instance_info = self.get_instance(instance_name, num_agents=num_agents)

        return instance_info

if __name__ == '__main__':

    number_instances = 128
    print('starting validation set generation')

    # validation set generation
    for num_nodes, n_agent in [(51, 10)]:
        generator = InstanceGenerator(batch_size=64, seed=0)
        for k in range(number_instances):
            instance =  generator.sample_instance(num_agents=n_agent, num_nodes=num_nodes)
            name = f'generated_val_servs_{num_nodes-1}_agents_{n_agent}_{k}'
            instance['name'] = name
            if not os.path.exists(f'data/generated/val_servs_{num_nodes-1}_agents_{n_agent}'):
                os.makedirs(f'data/generated/val_servs_{num_nodes-1}_agents_{n_agent}')
            with open(f'data/generated/val_servs_{num_nodes-1}_agents_{n_agent}/'+name+'.pkl', 'wb') as fp:
                pickle.dump(instance, fp, protocol=pickle.HIGHEST_PROTOCOL)

    print('done')
