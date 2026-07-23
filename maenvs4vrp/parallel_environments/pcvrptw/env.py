import torch
from tensordict import TensorDict

from typing import Optional, Dict, List

import warnings

from maenvs4vrp.core.env_generator_builder import InstanceBuilder
from maenvs4vrp.core.env_observation_builder import ObservationBuilder
from maenvs4vrp.core.env_agent_reward import RewardFn
from maenvs4vrp.core.parallel_env import PEnv
from maenvs4vrp.utils.ops import gather_by_index, get_distance


class Environment(PEnv):
    """
    PCVRPTW parallel environment class.
    """
    env_name = 'pcvrptw'

    def __init__(self,
                instance_generator_object: InstanceBuilder,
                obs_builder_object: ObservationBuilder,
                reward_evaluator: RewardFn,
                seed=None,
                device: Optional[str] = None,
                batch_size: torch.Size = None):
        """
        Constructor.

        Args:
            instance_generator_object(InstanceBuilder): Generator instance.
            obs_builder_object(ObservationBuilder): Observations instance.
            reward_evaluator(RewardFn): Reward evaluator instance.
            seed(int): Random number generator seed. Defaults to None.
            device(str, optional): Type of processing. It can be "cpu" or "gpu". Defaults to None.
            batch_size(torch.Size): Batch size. Defaults to None.
        """

        self.version = 'v0'

        # seed the environment
        if seed is None:
            self._set_seed(self.DEFAULT_SEED)
        else:
            self._set_seed(seed)

        self.inst_generator = instance_generator_object
        self.inst_generator._set_seed(self.seed)
        self.obs_builder = obs_builder_object
        self.obs_builder.set_env(self)
        self.reward_evaluator = reward_evaluator
        self.reward_evaluator.set_env(self)
        self.env_nsteps = 0

        if device is None:
            self.device = self.inst_generator.device
        else:
            self.device = device
            self.inst_generator.device = device

        if batch_size is None:
            self.batch_size = self.inst_generator.batch_size
        else:
            batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
            self.batch_size = torch.Size(batch_size)
            self.inst_generator.batch_size = torch.Size(batch_size)

        self.td_state = TensorDict({}, batch_size=self.batch_size, device=self.device)


    def observe(self, td: TensorDict, obs_list=None)-> TensorDict:
        """
        Retrieve agent environment observations.

        Args:
            is_reset(bool): If the environment is on reset. Defauts to False.

        Returns
            td_observations(TensorDict): Current agent observaions and masks dictionary.
        """

        td_observations = self.obs_builder.get_observations(obs_list=obs_list)

        if obs_list is not None and 'active_agents_mask' in obs_list:
            td_observations['active_agents_mask'] = self.td_state['agents']['active_agents_mask'].clone()
        if obs_list is not None and 'agents_action_mask' in obs_list:
            self._update_all_agents_feasibility()
            td_observations['agents_action_mask'] = self.td_state['agents']['action_mask'].clone()
        if obs_list is not None and 'agents_cur_node_idx' in obs_list:
            td_observations['agents_cur_node_idx'] = self.td_state['agents']['cur_node_idx'].clone()

        td['observations'] = td_observations
        return td


    def sample_actions_all(self, td: TensorDict) -> TensorDict:
        """
        Sample random actions for ALL agents simultaneously without conflicts.

        No two active agents will be assigned the same non-depot node.
        Agents are processed in order (0 → A-1): after each agent samples,
        its chosen node is removed from all subsequent agents' masks
        (depot can still be chosen by multiple agents).
        Inactive agents (empty mask) are forced to return to the depot.

        Args:
            td(TensorDict): Environment tensor instance.

        Returns:
            td(TensorDict): Updated tensor with ``"next_actions"`` of shape [B, A].
        """
        batch_size = self.td_state.batch_size
        depot_idx = self.td_state['depot_idx']       # [B, 1]
        depot_idx_1d = depot_idx.squeeze(-1)         # [B]

        mask = self.td_state['agents']['action_mask'].clone().float()  # [B, A, N]
        actions = torch.zeros(*batch_size, self.num_agents, dtype=torch.int64, device=self.device)

        for a in range(self.num_agents):
            agent_mask = mask[:, a, :]   # [B, N]

            # Inactive agents — fall back to the depot
            no_feasible = agent_mask.sum(dim=-1, keepdim=True).eq(0)  # [B, 1]
            if no_feasible.any():
                depot_col = torch.zeros_like(agent_mask)
                depot_col.scatter_(1, depot_idx, 1.0)
                agent_mask = torch.where(no_feasible.expand_as(agent_mask), depot_col, agent_mask)

            action_a = torch.multinomial(agent_mask, 1).squeeze(-1)   # [B]
            actions[:, a] = action_a

            # Block the chosen node for all remaining agents (depot stays open)
            non_depot = ~action_a.eq(depot_idx_1d)   # [B]
            if non_depot.any():
                conflict = torch.zeros(*batch_size, self.num_nodes, device=self.device)
                conflict.scatter_(1, action_a.unsqueeze(-1), non_depot.unsqueeze(-1).float())
                mask = mask * (1.0 - conflict.unsqueeze(1))

        # For already-done batches all agents go to the depot
        done = self.td_state['done']   # [B]
        if done.any():
            depot_actions = depot_idx.expand(*batch_size, self.num_agents)  # [B, A]
            actions = torch.where(done.unsqueeze(-1).expand_as(actions), depot_actions, actions)

        td['next_actions'] = actions
        return td


    def reset(self,
              num_agents:int|None=None,
              num_nodes:int|None=None,
              capacity:int|None=None,
              service_times:float|None=None,
              profits:str='constant',
              speed:float=None,
              instance_name:str|None=None,
              sample_type:str='random',
              instance_dict:Dict=None,
              force_visit: bool = False,
              batch_size: Optional[torch.Size] = None,
              n_augment: Optional[int] = None,
              seed:int|None=None,
              device: Optional[str] = "cpu")-> TensorDict:
        """
        Reset the environment.

        Args:
            num_agents(int, optional): Total number of agents. Defaults to None.
            num_nodes(int, optional): Total number of nodes. Defaults to None.
            capacity(int, optional): Total capacity for each agent. Defaults to None.
            service_times(float, optional): Service time in the nodes. Defaults to None.
            speed (float): Travel speed for all agents. Defaults to None.
            instance_name(str, optional): Instance name. Defaults to None.
            sample_type(str): Sample type. It can be "random", "augment" or "saved". Defaults to "random".
            instance_dict(Dict, optional): Instance dictionary. Defaults to None.
            force_visit(bool, optional): Force visit. Defaults to False.
            batch_size(torch.Size, optional): Batch size. Defaults to None.
            n_augment(int, optional): Data augmentation. Defaults to None.
            seed(int, optional): Random number generator seed. Defaults to None.

        Returns:
            TensorDict: Environment information dictionary.
        """

        if seed is not None:
            self._set_seed(seed)

        if batch_size is None:
            batch_size = self.batch_size
        else:
            batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
            self.batch_size = torch.Size(batch_size)
            self.inst_generator.batch_size = torch.Size(batch_size)

        if force_visit is not None:
            self.force_visit = force_visit

        instance_info = self.inst_generator.sample_instance(num_agents=num_agents,
                                                            num_nodes=num_nodes,
                                                            capacity=capacity,
                                                            service_times=service_times,
                                                            profits=profits,
                                                            speed=speed,
                                                            instance_name=instance_name,
                                                            sample_type=sample_type,
                                                            batch_size=batch_size,
                                                            n_augment=n_augment,
                                                            seed=seed,
                                                            device=device)

        self.num_nodes = instance_info['num_nodes']
        self.num_agents = instance_info['num_agents']

        if 'n_digits' in instance_info:
            self.n_digits = instance_info['n_digits']
        else:
            self.n_digits = None

        self.td_state = instance_info['data']
        self.td_state['speed'] = instance_info['data']['speed'].clone()

        self.td_state['done'] = torch.zeros(*batch_size, dtype=torch.bool)
        self.td_state['is_last_step'] = torch.zeros(*batch_size, dtype=torch.bool)
        self.td_state['depot_loc'] = self.td_state['coords'].gather(1, self.td_state['depot_idx'][:,:,None].expand(-1, -1, 2))

        self.td_state['max_tour_duration'] =  self.td_state['end_time'] - self.td_state['start_time']

        distance2depot = get_distance(self.td_state['depot_loc'], self.td_state['coords'])
        time2depot = distance2depot / self.td_state['speed']
        if self.n_digits is not None:
            distance2depot = torch.floor(self.n_digits * distance2depot) / self.n_digits
            time2depot = torch.floor(self.n_digits * time2depot) / self.n_digits

        self.td_state['distance2depot'] = distance2depot
        self.td_state['time2depot'] = time2depot

        self.td_state['nodes'] = TensorDict(
                                    source={'cur_profits': self.td_state['profits'].clone(),
                                            'cur_demands': self.td_state['demands'].clone(),
                                            'distance2depot': distance2depot,
                                            'time2depot': time2depot,
                                            'active_nodes_mask': torch.ones((*batch_size, self.num_nodes),dtype=torch.bool, device=self.device)},
                                    batch_size=batch_size, device=self.device)
        self.td_state['agents'] =  TensorDict(
                                    source={'cum_profit': torch.zeros((*batch_size, self.num_agents), dtype = torch.float, device=self.device),
                                            'cur_profit': torch.zeros((*batch_size, self.num_agents), dtype = torch.float, device=self.device),
                                            'capacity': self.td_state['capacity'],
                                            'cur_load': self.td_state['capacity'].clone() * torch.ones((*batch_size, self.num_agents), dtype = torch.float, device=self.device),
                                            'cur_time': self.td_state['start_time'].unsqueeze(1).clone() * torch.ones((*batch_size, self.num_agents), dtype = torch.float, device=self.device),
                                            'cur_node_idx': self.td_state['depot_idx'] * torch.ones((*batch_size, self.num_agents), dtype = torch.int64, device=self.device),
                                            'cur_travel_time': torch.zeros((*batch_size, self.num_agents), dtype = torch.float, device=self.device),
                                            'cum_travel_time': torch.zeros((*batch_size, self.num_agents), dtype = torch.float, device=self.device),
                                            'visited_nodes': torch.zeros((*batch_size, self.num_agents, self.num_nodes), dtype=torch.bool, device=self.device),
                                            'action_mask': torch.ones((*batch_size, self.num_agents, self.num_nodes), dtype=torch.bool, device=self.device),
                                            'active_agents_mask': torch.ones((*batch_size, self.num_agents), dtype=torch.bool, device=self.device),
                                            'cur_step': torch.zeros((*batch_size, self.num_agents), dtype=torch.int32, device=self.device)},
                                    batch_size=batch_size, device=self.device)

        self.td_state['solution'] = TensorDict({}, batch_size=batch_size)

        self.obs_builder.set_env(self)
        self.reward_evaluator.set_env(self)

        done = self.td_state['done'].clone()
        reward = torch.zeros_like(done, dtype=torch.float, device=self.device)
        penalty = torch.zeros_like(done, dtype=torch.float, device=self.device)

        self.env_nsteps = 0
        self._update_all_agents_feasibility()
        return TensorDict(
            {
                "reward": reward,
                "penalty": penalty,
                "done": done,
                "cur_node_idx": self.td_state['agents']['cur_node_idx'].clone(),
            },
            batch_size=batch_size, device=self.device)

    def reset_observe(self,
              num_agents:int|None=None,
              num_nodes:int|None=None,
              capacity:float|None=None,
              service_times:float|None=None,
              profits:str='constant',
              speed:float|None=None,
              instance_name:str|None=None,
              sample_type:str='random',
              instance_dict:Dict=None,
              force_visit: bool = False,
              batch_size: Optional[torch.Size] = None,
              n_augment: Optional[int] = None,
              seed:int|None=None,
              device: Optional[str] = "cpu",
              obs_list: Optional[List[str]] = ['agents_action_mask']) -> TensorDict:
        """
        Resets and observe the environment.

        Args:
            num_agents(int, optional): Total number of agents. Defaults to None.
            num_nodes(int, optional): Total number of nodes. Defaults to None.
            capacity(float, optional): Total capacity for each agent. Defaults to None.
            service_times(float, optional): Total service times for each agent. Defaults to None.
            profits(str, optional): Profit strategy. Defaults to 'constant'.
            speed(float, optional): Vehicles' speed. Defaults to None.
            instance_name(str, optional): Instance name. Defaults to None.
            sample_type(str): Sample type. It can be "random", "augment" or "saved". Defaults to "random".
            force_visit(bool): It forces the agent to visit all feasible nodes before going back to depot. Defaults to True.
            batch_size(torch.Size, optional): Batch size. Defaults to None.
            n_augment(int, optional): Data augmentation. Defaults to None.
            seed(int, optional): Random number generator seed. Defaults to None.
            obs_list(List[str], optional): List of observations to be retrieved. Defaults to ['agents_action_mask'].

        Returns:
            TensorDict: Environment information dictionary.
        """

        td = self.reset(num_agents=num_agents,
                            num_nodes=num_nodes,
                            capacity=capacity,
                            service_times=service_times,
                            profits=profits,
                            speed=speed,
                            instance_name=instance_name,
                            sample_type=sample_type,
                            instance_dict=instance_dict,
                            force_visit=force_visit,
                            batch_size=batch_size,
                            n_augment=n_augment,
                            seed=seed,
                            device=device)

        td = self.observe(td, obs_list)
        return td

    def _update_all_agents_feasibility(self):
        """
        Update actions feasibility for all agents simultaneously.
        """
        # active_nodes_mask: [B, num_nodes] -> [B, num_agents, num_nodes]
        _mask = self.td_state['nodes']['active_nodes_mask'].unsqueeze(1).expand(-1, self.num_agents, -1).clone()

        # cur_node_idx: [B, num_agents] -> locs: [B, num_agents, 2]
        cur_node_idx = self.td_state['agents']['cur_node_idx']
        locs = self.td_state['coords'].gather(
            1, cur_node_idx.unsqueeze(-1).expand(-1, -1, 2)
        )  # [B, num_agents, 2]

        # distance/time from each agent to each node: [B, num_agents, num_nodes]
        locs_exp   = locs.unsqueeze(2).expand(-1, -1, self.num_nodes, -1)
        coords_exp = self.td_state['coords'].unsqueeze(1).expand(-1, self.num_agents, -1, -1)
        distance2j = torch.norm(locs_exp - coords_exp, dim=-1)   # [B, num_agents, num_nodes]
        time2j     = distance2j / self.td_state['speed'].unsqueeze(1)  # [B, num_agents, num_nodes]

        if self.n_digits is not None:
            distance2j = torch.floor(self.n_digits * distance2j) / self.n_digits
            time2j     = torch.floor(self.n_digits * time2j)     / self.n_digits

        # ptime: [B, num_agents] -> [B, num_agents, 1]
        ptime = self.td_state['agents']['cur_time'].unsqueeze(-1)

        arrivej        = ptime + time2j                                        # [B, num_agents, num_nodes]
        waitj          = torch.clip(self.td_state['tw_low'].unsqueeze(1) - arrivej, min=0)
        service_startj = arrivej + waitj

        # same constraints as _update_curr_agent_feasibility
        c1 = service_startj <= self.td_state['tw_high'].unsqueeze(1)           # [B, num_agents, num_nodes]
        c2 = (service_startj +
              self.td_state['service_time'].unsqueeze(1) +
              self.td_state['time2depot'].unsqueeze(1)) <= self.td_state['end_time'].unsqueeze(-1).unsqueeze(-1)

        # capacity: demands [B, num_nodes], cur_load [B, num_agents]
        c3 = self.td_state['demands'].unsqueeze(1) <= self.td_state['agents']['cur_load'].unsqueeze(-1)

        _mask = _mask * c1 * c2 * c3

        _mask = self._post_process_mask(_mask)
        self.td_state['agents']['action_mask'] = _mask


    def _post_process_mask(self, mask):
        """
        Post-process the action mask after all constraints have been applied.
        No force_visit for PCVRPTW (profit maximization, visits are optional).
        """
        batch_size = self.td_state.batch_size
        active_agents = self.td_state['agents']['active_agents_mask']          # [B, num_agents]

        # Depot must always be open for active agents
        depot_idx_exp = self.td_state['depot_idx'].unsqueeze(1).expand(*batch_size, self.num_agents, 1)  # [B, A, 1]
        mask.scatter_(2, depot_idx_exp, active_agents.unsqueeze(-1))

        # Zero out inactive agents
        active_expanded = active_agents.unsqueeze(-1).expand(-1, -1, self.num_nodes)
        mask = mask & active_expanded

        # After done, close all
        done = self.td_state['done']                                           # [B]
        mask = mask & ~done.unsqueeze(-1).unsqueeze(-1)

        # Open depot for agent 0 only in done rows (batch consistency)
        done_rows = done.squeeze(-1).nonzero(as_tuple=True)[0]
        if done_rows.numel() > 0:
            depot_idx = self.td_state['depot_idx'].squeeze(-1) if self.td_state['depot_idx'].dim() > 1 else self.td_state['depot_idx']
            mask[done_rows, 0, depot_idx[done_rows]] = True

        return mask

    def _update_done(self, actions):
        """
        Update done state for a single parallel step (all agents).

        Args:
            actions(torch.Tensor): [B, A] tensor of chosen node indices.

        Returns:
            None.
        """
        former_done = self.td_state['done'].clone()
        depot_idx = self.td_state['depot_idx']   # [B, 1]

        # An agent is deactivated when it returns to depot
        went_to_depot = actions.eq(depot_idx)    # [B, A]
        self.td_state['agents']['active_agents_mask'] = (
            self.td_state['agents']['active_agents_mask'] & ~went_to_depot
        )

        self.td_state['done'] = (~self.td_state['agents']['active_agents_mask']).all(dim=-1)
        self.td_state['done'] = self.td_state['done'] | former_done

        # Deactivate visited non-depot nodes
        depot_idx_exp = depot_idx.unsqueeze(1).expand(-1, self.num_agents, -1)   # [B, A, 1]
        non_depot_actions = torch.where(
            actions.unsqueeze(-1).eq(depot_idx_exp),
            depot_idx.unsqueeze(1).expand(-1, self.num_agents, -1),
            actions.unsqueeze(-1)
        ).squeeze(-1)    # [B, A]
        # scatter False for each visited non-depot node
        for a in range(self.num_agents):
            not_depot = ~actions[:, a].eq(depot_idx.squeeze(-1))   # [B]
            if not_depot.any():
                self.td_state['nodes']['active_nodes_mask'].scatter_(
                    1, actions[:, a].unsqueeze(-1),
                    ~not_depot.unsqueeze(-1)
                )

        self.td_state['is_last_step'] = self.td_state['done'] & ~former_done

    def _update_done_all(self, actions):
        """Alias for _update_done (parallel step)."""
        self._update_done(actions)


    def _update_state(self, actions):
        """
        Update environment state for all agents simultaneously (parallel step).

        Args:
            actions(torch.Tensor): [B, A] chosen node indices.

        Returns:
            None.
        """
        agents = self.td_state['agents']
        depot_idx = self.td_state['depot_idx']      # [B, 1]

        # Current locations: [B, A, 2]
        cur_node_idx = agents['cur_node_idx']        # [B, A]
        locs = self.td_state['coords'].gather(
            1, cur_node_idx.unsqueeze(-1).expand(-1, -1, 2)
        )
        # Next locations: [B, A, 2]
        next_locs = self.td_state['coords'].gather(
            1, actions.unsqueeze(-1).expand(-1, -1, 2)
        )

        ptime = agents['cur_time'].clone()           # [B, A]

        # Travel time: [B, A]
        distance2j = torch.norm(locs - next_locs, dim=-1)
        time2j = distance2j / self.td_state['speed']
        if self.n_digits is not None:
            distance2j = torch.floor(self.n_digits * distance2j) / self.n_digits
            time2j = torch.floor(self.n_digits * time2j) / self.n_digits

        # TW waiting
        tw_low_j = self.td_state['tw_low'].gather(1, actions)    # [B, A]
        service_time_j = self.td_state['service_time'].gather(1, actions)  # [B, A]
        arrivej = ptime + time2j
        waitj = torch.clip(tw_low_j - arrivej, min=0)
        time_update = arrivej + waitj + service_time_j

        # Agents that are now inactive (just returned to depot)
        agents_done = ~agents['active_agents_mask']  # [B, A]

        # Update cur_node_idx
        agents['cur_node_idx'] = actions

        # Update cur_time (done agents → end_time)
        end_time = self.td_state['end_time'].unsqueeze(-1).expand_as(time_update)  # [B, A]
        agents['cur_time'] = torch.where(agents_done, end_time, time_update)

        # Update travel time accumulators
        agents['cur_travel_time'] = time2j
        agents['cum_travel_time'] = agents['cum_travel_time'] + time2j

        # Update load: decrease by demands of visited node; depot visit resets to capacity
        demands_taken = self.td_state['demands'].gather(1, actions)  # [B, A]
        went_to_depot = actions.eq(depot_idx)                        # [B, A]
        new_load = agents['cur_load'] - demands_taken
        # Restore capacity on depot visit (but done agents get 0)
        new_load = torch.where(went_to_depot, agents['capacity'].expand_as(new_load), new_load)
        new_load = torch.where(agents_done, torch.zeros_like(new_load), new_load)
        agents['cur_load'] = new_load

        # Zero demands in visited non-depot nodes
        self.td_state['nodes']['cur_demands'].scatter_(
            1, actions,
            torch.where(went_to_depot,
                        self.td_state['nodes']['cur_demands'].gather(1, actions),
                        torch.zeros_like(actions, dtype=torch.float))
        )

        # Update visited nodes
        non_depot = ~went_to_depot   # [B, A]
        for a in range(self.num_agents):
            row_mask = non_depot[:, a]   # [B]
            agents['visited_nodes'][row_mask, a, actions[row_mask, a]] = True

        # Update profits
        profits_taken = self.td_state['profits'].gather(1, actions)  # [B, A]
        agents['cum_profit'] = agents['cum_profit'] + profits_taken
        agents['cur_profit'] = profits_taken
        # Zero out profits of visited non-depot nodes
        self.td_state['nodes']['cur_profits'].scatter_(
            1, actions,
            torch.where(went_to_depot,
                        self.td_state['nodes']['cur_profits'].gather(1, actions),
                        torch.zeros_like(actions, dtype=torch.float))
        )

        # Update step counter
        agents['cur_step'] = torch.where(
            ~agents_done,
            agents['cur_step'] + 1,
            agents['cur_step']
        )

        # Batch consistency: if all done, re-activate agent 0
        self.td_state['agents']['active_agents_mask'][
            self.td_state['agents']['active_agents_mask'].sum(1).eq(0), 0
        ] = True

    def _update_state_all(self, actions):
        """Alias for _update_state (parallel step)."""
        self._update_state(actions)

    def _update_solution_all(self, actions):
        """
        Update solution with all-agent parallel actions.

        Args:
            actions(torch.Tensor): [B, A] chosen node indices.

        Returns:
            None.
        """
        agent_indices = torch.arange(self.num_agents, device=self.device)\
            .unsqueeze(0).expand(*self.batch_size, -1)  # [B, A]

        if 'actions' in self.td_state['solution'].keys():
            self.td_state['solution', 'actions'] = torch.cat(
                [self.td_state['solution', 'actions'], actions], dim=-1
            )
            self.td_state['solution', 'agents'] = torch.cat(
                [self.td_state['solution', 'agents'], agent_indices], dim=-1
            )
        else:
            self.td_state['solution', 'actions'] = actions
            self.td_state['solution', 'agents'] = agent_indices

    def step_all(self, td: TensorDict) -> TensorDict:
        """
        Perform a parallel environment step for all agents simultaneously.

        Args:
            td(TensorDict): Environment tensor instance with ``"next_actions"`` [B, A].

        Returns:
            td(TensorDict): Updated environment tensor instance.
        """
        actions = td['next_actions']   # [B, A]

        self._update_done_all(actions)
        done = self.td_state['done'].clone()
        is_last_step = self.td_state['is_last_step'].clone()

        self._update_state_all(actions)
        self._update_solution_all(actions)

        reward, penalty = self.reward_evaluator.get_reward(actions)

        self.env_nsteps += 1
        td.update(
            {
                "reward": reward,
                "penalty": penalty,
                "done": done,
                "is_last_step": is_last_step,
                "cur_node_idx": self.td_state['agents']['cur_node_idx'].clone(),
            }
        )
        return td

    def step_all_observe(self, td: TensorDict,
                         obs_list: Optional[List[str]] = ['agents_action_mask']) -> TensorDict:
        """
        Parallel step followed by observations.

        Args:
            td(TensorDict): Environment tensor instance.
            obs_list(List[str], optional): Observation keys. Defaults to ['agents_action_mask'].

        Returns:
            td(TensorDict): Updated environment tensor instance.
        """
        td = self.step_all(td)
        td = self.observe(td, obs_list=obs_list)
        return td

    def step_observe(self, td: TensorDict,
                     obs_list: Optional[List[str]] = ['agents_action_mask']) -> TensorDict:
        """
        Parallel step followed by observations (alias for step_all_observe).

        Args:
            td(TensorDict): Environment tensor instance.
            obs_list(List[str], optional): Observation keys. Defaults to ['agents_action_mask'].

        Returns:
            td(TensorDict): Updated environment tensor instance.
        """
        return self.step_all_observe(td, obs_list=obs_list)

    def check_solution_validity(self):
        """
        Check if solution is valid according to constraints.

        Args:
            N/a.

        Returns:
            None.
        """

        distance2depot = get_distance(self.td_state['coords'], self.td_state['coords'][..., 0:1, :])
        time2depot = distance2depot / self.td_state['speed']
        a = self.td_state['tw_low'] + time2depot + self.td_state['service_time']  # Time to serve node and get back to depot
        b = self.td_state['tw_high'][:, 0, None]  # Depot late tw

        # Can agent serve node and get back to depot?
        assert torch.all(a <= b), "Agent cannot serve node and get back to depot."

        curr_node = torch.zeros(*self.batch_size, dtype=torch.int64, device=self.device)
        curr_time = torch.zeros(*self.batch_size, dtype=torch.float32, device=self.device)
        visited_nodes = torch.zeros(*self.batch_size, self.num_nodes, dtype=torch.int64, device=self.device)

        # Sort indices along each row
        sorted_indices = torch.argsort(self.td_state['solution']['agents'], dim=-1, stable=True)
        # Use gather to reorder data per row
        sorted_data = torch.gather(self.td_state['solution']['actions'], dim=-1, index=sorted_indices)

        demand = self.td_state['demands'].gather(1, sorted_data)

        for ii in range(sorted_data.size(1)):
            next_node = sorted_data[:, ii]
            curr_loc = gather_by_index(self.td_state['coords'], curr_node)
            next_loc = gather_by_index(self.td_state['coords'], next_node)

            dist = get_distance(curr_loc, next_loc)
            time = dist / self.td_state['speed'].squeeze(-1)
            if self.n_digits is not None:
                dist = torch.floor(self.n_digits * dist) / self.n_digits
                time = torch.floor(self.n_digits * time) / self.n_digits

            fill = visited_nodes.gather(1, next_node.unsqueeze(-1))
            visited_nodes.scatter_(1, next_node.unsqueeze(-1), fill + 1)

            curr_time = torch.max(curr_time + dist, gather_by_index(self.td_state['tw_low'], next_node))
            assert torch.all(curr_time <= gather_by_index(self.td_state['tw_high'], next_node)), "Agent must perform service before node's time window closes."

            curr_time = curr_time + gather_by_index(self.td_state['service_time'], next_node)
            curr_node = next_node
            curr_time[next_node == 0] = 0.0

        visited_nodes_exc_depot = visited_nodes[:, 1:]
        assert torch.all((visited_nodes_exc_depot == 0) | (visited_nodes_exc_depot == 1)), "Nodes were visited more than once!"

        # c3: capacity constraint
        used_cap = torch.zeros_like(self.td_state['demands'][:, 0])
        for ii in range(sorted_data.size(1)):
            #reset at depot
            used_cap = used_cap * (sorted_data[:, ii] != 0)
            used_cap += demand[:, ii]

            #Loads must no exceed capacity
            assert torch.all(used_cap <= self.td_state['capacity']), "Agent exceeded vehicle capacity."
