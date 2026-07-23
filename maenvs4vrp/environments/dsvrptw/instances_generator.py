import torch
from tensordict import TensorDict

import os
from os import path
import pickle

from typing import Dict, Optional
from maenvs4vrp.core.env_generator_builder import InstanceBuilder

import warnings
import shutil
import random
from huggingface_hub import HfApi, snapshot_download

HF_REPO_ID = "maenvs4vrp/environments"
INSTANCES_PATH = 'dsvrptw/data/generated'
DATA_PATH = './dsvrptw/data/generated'


class InstanceGenerator(InstanceBuilder):
    """
    DSVRPTW instance generation class.
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
                 num_agents:Optional[int]=25,
                 num_nodes:Optional[int]=100,
                 capacity:Optional[int]=200,
                 min_cust_count:Optional[int] = None,
                 cust_loc_range:Optional[tuple] = (0,101),
                 cust_dem_range:Optional[tuple]= (5,41),
                 horizon:Optional[int] = 480,
                 service_times_range:Optional[tuple] = (10,31),
                 tw_ratio:Optional[float] = 0.5,
                 cust_tw_range:Optional[tuple] = (30,91),
                 dod:Optional[float] = 0.5,
                 d_early_ratio:Optional[float]= 0.5,
                 instance_name:Optional[str]='validation',
                 list_of_instances:Optional[set]=None,
                 device: Optional[str] = "cpu",
                 batch_size: Optional[torch.Size] = None,
                 seed:Optional[int]=None) -> None:
        """
        Constructor. Instance generator.

        Args:
            num_agents(int): Total number of agents. Defaults to 25.
            num_nodes(int):  Total number of nodes. Defaults to 100.
            capacity(int): Agent capacity. Defaults to 200.
            min_cust_count(int): Minimum number of customers. Defaults to None.
            cust_loc_range(tuple): Customer location range. Defaults to (0,101).
            cust_dem_range(tuple): Customer demand range. Defaults to (5,41).
            horizon(int): Time horizon. Defaults to 480.
            service_times_range(tuple): Service times range. Defaults to (10,31).
            tw_ratio(float): Time windows ratio. Defaults to 0.5.
            cust_tw_range(tuple): Customer time windows range. Defaults to (30,91).
            dod(float): Dynamic customers ratio. Defaults to 0.5.
            d_early_ratio(float): Early dynamic customers ratio. Defaults to 0.5.
            instance_name(str): Instance name. Can be "validation" or "test". Defaults to "validation".
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

        self.device = device
        if batch_size is None:
            batch_size = [1]
        else:
            batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        self.batch_size = torch.Size(batch_size)

        if num_agents is not None:
            assert num_agents>0, f"number of agents must be grater them 0!"
            self.num_agents = num_agents
        if num_nodes is not None:
            assert num_nodes>0, f"number of services must be grater them 0!"
            self.num_nodes = num_nodes
        if service_times_range is not None:
            self.service_times_range = service_times_range
        if capacity is not None:
            assert capacity>0, f"agent capacity must be grater them 0!"
            self.capacity = capacity

        self.min_cust_count = min_cust_count
        self.cust_loc_range = cust_loc_range
        self.cust_dem_range = cust_dem_range
        self.horizon = horizon
        self.tw_ratio = tw_ratio
        self.cust_tw_range = cust_tw_range
        self.dod = dod
        self.d_early_ratio = d_early_ratio

        assert instance_name in ["validation", "test"], f"instance unknown type"
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



    def read_instance_data(self, instance_name:Optional[str])-> Dict:
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


    def get_instance(self, instance_name:Optional[str], num_agents:Optional[int]=None) -> Dict:

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



    def random_generate_instance(self,
                                 batch_size: Optional[torch.Size] = None,
                                 seed:Optional[int]=None,
                                 device:Optional[str]="cpu")-> TensorDict:
        """
        Generate random instance.

        Follows https://gitlab.inria.fr/gbono/mardam/-/blob/master/problems/_data_sdtw.py

        Args:
            batch_size (Optional[torch.Size]): Batch size. Defaults to None.
            seed (int, optional): Random number generator seed. Defaults to None.

        Returns:
            TensorDict: Instance data.
        """
        if seed is not None:
            self._set_seed(seed)


        if batch_size is not None:
            batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
            self.batch_size = torch.Size(batch_size)

        size = (*self.batch_size, self.num_nodes)

        instance = TensorDict({}, batch_size=self.batch_size, device=self.device)

        self.depot_idx = 0
        instance['depot_idx'] = self.depot_idx * torch.ones((*self.batch_size, 1), dtype = torch.int64, device=self.device)

        # Sample coords        x_j, y_j ~ U(0, 100)
        locs = torch.randint(*self.cust_loc_range, (*self.batch_size, self.num_nodes, 2), dtype = torch.float)
        # Sample demands             q_j ~ U(5,  40)
        demands = torch.randint(*self.cust_dem_range, (*self.batch_size, self.num_nodes), dtype = torch.float)
        demands[:, self.depot_idx] = 0.0

        # Sample service_times       s_j ~ U(10, 30)
        service_times = torch.randint(*self.service_times_range, (*self.batch_size, self.num_nodes), dtype = torch.float)
        service_times[:, self.depot_idx] = 0

        instance['coords'] = locs
        instance['demands'] = demands
        instance['service_time'] = service_times

        # Sample dyn subset           ~ B(dod)
        # and early/late appearance   ~ B(d_early_ratio)
        if isinstance(self.dod, float):
            is_dyn = torch.empty(size).bernoulli_(self.dod)
        elif len(self.dod) == 1:
            is_dyn = torch.empty(size).bernoulli_(self.dod[0])
        else: # tuple of float
            ratio = torch.tensor(self.dod)[torch.randint(0, len(self.dod), (batch_size,), dtype = torch.int64)]
            is_dyn = ratio[:,None,None].expand(*size).bernoulli()
        is_dyn[:, self.depot_idx] = 0

        if isinstance(self.d_early_ratio, float):
            is_dyn_e = torch.empty(size).bernoulli_(self.d_early_ratio)
        elif len(self.d_early_ratio) == 1:
            is_dyn_e = torch.empty(size).bernoulli_(self.d_early_ratio[0])
        else:
            ratio = torch.tensor(self.d_early_ratio)[
                    torch.randint(0, len(self.d_early_ratio), (batch_size,), dtype = torch.int64)
                    ]
            is_dyn_e = ratio[:,None,None].expand(*size).bernoulli()
        is_dyn_e[:, self.depot_idx] = 0

        # Sample appear. time     a_j = 0 if not in D subset
        #                         a_j ~ U(1,H/3) if early appear
        #                         a_j ~ U(H/3+1, 2H/3) if late appear
        aprs = is_dyn * is_dyn_e * torch.randint(1, self.horizon//3+1, size, dtype = torch.float) \
                + is_dyn * (1-is_dyn_e) * torch.randint(self.horizon//3+1, 2*self.horizon//3+1, size, dtype = torch.float)
        aprs[:, self.depot_idx] = 0


        # Sample TW subset            ~ B(tw_ratio)
        if isinstance(self.tw_ratio, float):
            has_tw = torch.empty(size).bernoulli_(self.tw_ratio)
        elif len(self.tw_ratio) == 1:
            has_tw = torch.empty(size).bernoulli_(self.tw_ratio[0])
        else: # tuple of float
            ratio = torch.tensor(self.tw_ratio)[torch.randint(0, len(self.tw_ratio), (batch_size,), dtype = torch.int64)]
            has_tw = ratio[:,None,None].expand(*size).bernoulli()

        # Sample TW width        tw_j = H if not in TW subset
        #                        tw_j ~ U(30,90) if in TW subset
        tws = (1 - has_tw) * torch.full(size, self.horizon) \
                + has_tw * torch.randint(*self.cust_tw_range, size, dtype = torch.float)
        tws[:, self.depot_idx] = self.horizon

        # Compute depot-to-customer travel times
        depot_locs = locs[:, self.depot_idx, :]  # shape: [batch_size, 2]
        cust_locs = locs  # shape: [batch_size, num_nodes, 2]
        tt_0j = (depot_locs.unsqueeze(1) - cust_locs).pow(2).sum(-1).pow(0.5)  # [batch_size, num_nodes]

        # Sample ready time
        rdys = has_tw * (aprs + torch.rand(size) * (self.horizon - torch.max(tt_0j + service_times, tws) - aprs))
        rdys.floor_()

        instance['tw_low'] =  rdys
        instance['tw_high'] = rdys + tws

        instance['is_depot'] = torch.zeros((*self.batch_size, self.num_nodes), dtype=torch.bool, device=self.device)
        instance['is_depot'][:, self.depot_idx] = True

        instance['start_time'] = torch.zeros(*self.batch_size, dtype=torch.int64, device=self.device)
        instance['end_time'] = self.horizon * torch.ones(*self.batch_size, dtype=torch.int64, device=self.device)
        instance['capacity'] = self.capacity * torch.ones((*self.batch_size, 1), dtype = torch.float, device=self.device)
        instance['appear_time'] =  aprs

        instance_info = {'name':'random_instance',
                         'late_penalty' : 2,
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
                         'late_penalty' : 2,
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
                        num_agents: Optional[int] = None,
                        num_nodes: Optional[int] = None,
                        instance_name: Optional[str] = None,
                        sample_type: Optional[str] = 'random',
                        batch_size: Optional[torch.Size] = None,
                        n_augment: Optional[int] = None,
                        seed:Optional[int] = None,
                        device: Optional[str] = "cpu")-> Dict:
        """
        Sample one instance from instance space.

        Args:
            num_agents(int): Total number of agents. Defaults to None.
            num_nodes(int):  Total number of nodes. Defaults to None.
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


        if batch_size is not None:
            batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
            self.batch_size = torch.Size(batch_size)
        else:
            batch_size = self.batch_size

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
