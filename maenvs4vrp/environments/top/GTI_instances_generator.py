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
INSTANCES_PATH = 'top/data/benchmark'
DATA_PATH = './top/data/benchmark'

class GTIGenerator(InstanceBuilder):
    """
    class for TOP GTI benchmark instances generation

    """
    @classmethod
    def get_list_of_instances(cls):
        base_dir = path.dirname(path.dirname(path.abspath(__file__)))

        return {'GTI_20': [s.split('.')[0] for s in os.listdir(path.join(base_dir, INSTANCES_PATH, 'GTI')) if '20_L2_' in s],
                'GTI_50': [s.split('.')[0] for s in os.listdir(path.join(base_dir, INSTANCES_PATH, 'GTI')) if '50_L2_' in s],
                'GTI_100': [s.split('.')[0] for s in os.listdir(path.join(base_dir, INSTANCES_PATH, 'GTI')) if '100_L2_' in s],}

    def __init__(self,
                 instance_name:Optional[str] = 'GTI',
                 list_of_instances:Optional[list] = None,
                 device:Optional[str] = "cpu",
                 batch_size:Optional[torch.Size] = None,
                 seed:Optional[int] = None) -> None:
        """

        Args:
            instance_name(str, Optional): instance name. Defaults to "GTI"
            list_of_instances(list, Optional): List of instances file names
            device(str, optional): Type of processing. It can be "cpu" or "gpu". Defaults to "cpu".
            batch_size(torch.Size, optional): Batch size. If not specified, defaults to 1.
            seed (int): random number generator seed. Defaults to None;
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

        assert instance_name in ["GTI"], f"instance unknown type"
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
        Reads instance data
        Args:
            instance_name (str): instance file name.

        Returns:
            Dict: Instance data
        """

        base_dir = path.dirname(path.dirname(path.abspath(__file__)))
        path_to_file = path.join(base_dir, INSTANCES_PATH, self.instance_name)
        generated_file = '{path_to_generated_instances}/{instance}.pkl' \
                        .format(path_to_generated_instances=path_to_file,
                                instance=instance_name)
        with open(generated_file, 'rb') as fp:
            data = pickle.load(fp)
            data = [
                        {
                            'loc': torch.FloatTensor(loc),
                            'prize': torch.FloatTensor(prize),
                            'depot': torch.FloatTensor(depot),
                            'max_length': torch.tensor(length)
                        }
                        for depot, loc, prize, length in data
                    ]
        instance = self.parse_instance_data(data, instance_name)

        return instance


    def get_instance(self, instance_name:str, num_agents:Optional[int] = None) -> Dict:
        """
        Returns:
            Dict: Instance data

        """
        instance = self.instances_data.get(instance_name)

        if num_agents is not None:
            assert num_agents>0, f"number of agents must be grater them 0!"
            instance['num_agents'] = num_agents

        return instance

    def parse_instance_data(self, instance_data: list, instance_name:str) -> Dict:
        """
        Parse instance data into dict

        """
        instance = dict()
        instance['name'] = instance_name

        coords = []
        profits = []
        max_length = []
        self.service_times = 0.0
        for data in instance_data:
            coords.append(torch.cat((data['depot'][None, :], data['loc']), -2).unsqueeze(0))
            profits.append(torch.cat((torch.tensor([0]), data['prize']), -1).unsqueeze(0))
            max_length.append(data['max_length'])

        self.batch_size = torch.Size([len(instance_data)])
        data = TensorDict({}, batch_size=self.batch_size, device=self.device)

        self.depot_idx = 0
        data['depot_idx'] = self.depot_idx * torch.ones((*self.batch_size, 1), dtype = torch.int64, device=self.device)
        data['coords'] = torch.cat(coords, dim=0).to(self.device)

        num_nodes = data['coords'].shape[1]
        instance['num_agents'] = 4
        instance['num_nodes'] = num_nodes

        data['profits'] = torch.cat(profits, dim=0).to(self.device)

        service_times = self.service_times * torch.ones((*self.batch_size, num_nodes), dtype = torch.float, device=self.device)
        service_times[:, self.depot_idx] = 0
        data['service_time'] = service_times
        data['speed'] = torch.ones((*self.batch_size, 1), dtype=torch.float, device=self.device)

        data['start_time'] = torch.zeros((*self.batch_size, 1), device=self.device).squeeze(-1)

        data['end_time'] = torch.tensor(max_length, device=self.device)


        data['is_depot'] = torch.zeros((*self.batch_size, instance['num_nodes']), dtype=torch.bool, device=self.device)
        data['is_depot'][:, self.depot_idx] = True

        instance['data'] = data

        return instance


    def load_list_of_instances(self, list_of_instances:Optional[list] = None):
        """
        Loads every instance on List_of_instances List

        Args:
            List_of_instances(List, Optional):List of instances file names. Defaults to None.

        """
        if list_of_instances:
            self.list_of_instances = list_of_instances
        self.instances_data = dict()
        for instance_name in self.list_of_instances:
            instance = self.read_instance_data(instance_name)
            self.instances_data[instance_name] = instance


    def sample_name_from_list(self, seed:Optional[int] = None) -> str:
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
                        num_agents:Optional[int] = None,
                        num_nodes:Optional[int] = None,
                        service_times:Optional[float] = 0.0,
                        speed:Optional[float] = None,
                        profits:Optional[str] = 'constant',
                        instance_name:Optional[str] = None,
                        sample_type:Optional[str] = 'saved',
                        batch_size: Optional[torch.Size] = None,
                        n_augment: Optional[int] = None,
                        seed:Optional[int] = None) -> Dict:
        """
        Samples one instance from instance space

        Args:
            num_agents(int, Optional): Total number of agents. Defaults to 20.
            num_nodes(int, Optional):  Total number of nodes. Defaults to 100.
            service_times(float, Optional): Total time of service. Defaults to 0.2.
            speed(float, Optional): Vehicles' speed. Defaults to None.
            instance_name(str, Optional):  instance name. Defaults to None;
            sample_type(str): Sample type. Defaults to "saved".
            batch_size(torch.Size or None): Batch size. Defaults to None.
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


        if num_agents is None:
            num_agents = 4
        if num_nodes is None:
            num_nodes = 20
        if service_times is None:
            service_times = 0.0
        if speed is None:
            self.speed = 1.0
        else:
            self.speed = speed

        if batch_size is not None:
            batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
            self.batch_size = torch.Size(batch_size)

        if sample_type in ['random', 'augment']:
            raise NotImplementedError()
        elif sample_type=='saved':
            instance_info = self.get_instance(instance_name, num_agents=num_agents)

        return instance_info

if __name__ == '__main__':

    generator = GTIGenerator(instance_name="GTI", list_of_instances=["test_seed1234_const20_L2_0"])
    generator.sample_instance(num_agents=3, num_nodes=8, seed=1)
    print(generator)

    print('done')
