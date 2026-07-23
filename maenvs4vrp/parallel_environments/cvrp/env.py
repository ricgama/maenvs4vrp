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
    CVRP parallel environment generator class.
    """
    def __init__(self,
                instance_generator_object: InstanceBuilder,
                obs_builder_object: ObservationBuilder,
                reward_evaluator: RewardFn,
                seed=None,
                device: Optional[str] = None,
                batch_size: Optional[torch.Size] = None):

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
        self.env_name = 'cvrp'

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
            self.batch_size =  self.inst_generator.batch_size
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

        # Working copy so we can blank out claimed nodes mid-loop
        mask = self.td_state['agents']['action_mask'].clone().float()  # [B, A, N]

        actions = torch.zeros(*batch_size, self.num_agents, dtype=torch.int64, device=self.device)

        for a in range(self.num_agents):
            agent_mask = mask[:, a, :]   # [B, N]

            # Inactive agents have an all-zero mask — fall back to the depot
            no_feasible = agent_mask.sum(dim=-1, keepdim=True).eq(0)  # [B, 1]
            if no_feasible.any():
                depot_col = torch.zeros_like(agent_mask)
                depot_col.scatter_(1, depot_idx, 1.0)
                agent_mask = torch.where(no_feasible.expand_as(agent_mask), depot_col, agent_mask)

            action_a = torch.multinomial(agent_mask, 1).squeeze(-1)   # [B]
            actions[:, a] = action_a

            # Block the chosen node for all remaining agents, but only when
            # the choice is NOT the depot (depot stays available for everyone).
            non_depot = ~action_a.eq(depot_idx_1d)   # [B]
            if non_depot.any():
                # Build a [B, N] mask with 1 at the chosen node for non-depot rows
                conflict = torch.zeros(*batch_size, self.num_nodes, device=self.device)
                conflict.scatter_(1, action_a.unsqueeze(-1), non_depot.unsqueeze(-1).float())
                # Zero out that column for ALL agents (including future ones)
                mask = mask * (1.0 - conflict.unsqueeze(1))   # broadcast over A dim

        # For already-done batches all agents go to the depot (masks are invalid there)
        done = self.td_state['done']   # [B]
        if done.any():
            depot_actions = depot_idx.expand(*batch_size, self.num_agents)  # [B, A]
            actions = torch.where(
                done.unsqueeze(-1).expand_as(actions), depot_actions, actions
            )

        td['next_actions'] = actions
        return td

    def reset(self,
              num_agents:int|None=None,
              num_nodes:int|None=None,
              capacity:float|None=None,
              speed:float|None=None,
              instance_name:str|None=None,
              sample_type:str='random',
              instance_dict:Dict=None,
              force_visit: bool = False,
              batch_size: Optional[torch.Size] = None,
              n_augment: Optional[int] = None,
              seed:int|None=None,
              device: Optional[str] = "cpu")-> TensorDict:
        """
        Resets the environment.

        Args:
            num_agents(int, optional): Total number of agents. Defaults to None.
            num_nodes(int, optional): Total number of nodes. Defaults to None.
            capacity(float, optional): Total capacity for each agent. Defaults to None.
            speed(float, optional): Vehicles' speed. Defaults to None.
            instance_name(str, optional): Instance name. Defaults to None.
            sample_type(str): Sample type. It can be "random", "augment" or "saved". Defaults to "random".
            force_visit(bool): It forces the agent to visit all feasible nodes before going back to depot. Defaults to True.
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

        if instance_dict:
            instance_info = instance_dict
        else:
            instance_info = self.inst_generator.sample_instance(num_agents=num_agents,
                                                            num_nodes=num_nodes,
                                                            capacity=capacity,
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

        self.td_state['done'] = torch.zeros(*batch_size, dtype=torch.bool)
        self.td_state['is_last_step'] = torch.zeros(*batch_size, dtype=torch.bool)
        self.td_state['depot_loc'] = self.td_state['coords'].gather(1, self.td_state['depot_idx'][:,:,None].expand(-1, -1, 2))

        distance2depot = get_distance(self.td_state['depot_loc'], self.td_state['coords'])
        self.td_state['speed'] = instance_info['data']['speed'].clone()

        time2depot = distance2depot / self.td_state['speed']

        if self.n_digits is not None:
            distance2depot = torch.floor(self.n_digits * time2depot) / self.n_digits
            time2depot = torch.floor(self.n_digits * time2depot) / self.n_digits

        self.td_state['time2depot'] = time2depot

        self.td_state['nodes'] = TensorDict(
                                    source={'cur_demands': self.td_state['demands'].clone(),
                                            'active_nodes_mask': torch.ones((*batch_size, self.num_nodes),dtype=torch.bool, device=self.device),
                                            'not_visited_nodes_mask': torch.ones((*batch_size, self.num_nodes),dtype=torch.bool, device=self.device)},
                                    batch_size=batch_size, device=self.device)

        self.td_state['agents'] =  TensorDict(
                                    source={'capacity': self.td_state['capacity'],
                                            'cur_load': self.td_state['capacity'].clone() * torch.ones((*batch_size, self.num_agents), dtype = torch.float, device=self.device),
                                            'cur_time': torch.zeros((*batch_size, self.num_agents), dtype = torch.float, device=self.device),
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

        done = self.td_state['done']
        reward = torch.zeros_like(done, dtype = torch.float, device=self.device)
        penalty = torch.zeros_like(done, dtype = torch.float, device=self.device)

        self.env_nsteps = 0
        self._update_all_agents_feasibility()
        return TensorDict(
            {
                "reward": reward,
                "penalty":penalty,
                "done": done,
            },
            batch_size=batch_size, device=self.device)


    def reset_observe(self,
              num_agents:int|None=None,
              num_nodes:int|None=None,
              capacity:float|None=None,
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

        Args:
            n/a.

        Returns:
            None.
        """

        # Get dimensions
        batch_size = self.td_state.batch_size  # This should be a torch.Size object

        # Get base mask from active nodes
        _mask = self.td_state['nodes']['active_nodes_mask'].clone()  # [B, N]

        # Add agent dimension
        _mask = _mask.unsqueeze(1)  # [B, 1, N]
        _mask = _mask.expand(*batch_size, self.num_agents, self.num_nodes)  # [B, num_agents, N]

        # Capacity constraints
        # demands: [B, N] -> [B, 1, N] -> [B, num_agents, N]
        demands = self.td_state['demands'].unsqueeze(1).expand(*batch_size, self.num_agents, self.num_nodes)
        # cur_load: [B, num_agents] -> [B, num_agents, 1]
        cur_load = self.td_state['agents']['cur_load'].unsqueeze(-1)

        c1 = demands <= cur_load
        _mask = _mask & c1

        _mask = self._post_process_mask(_mask)

        self.td_state['agents']['action_mask'] = _mask

    def _post_process_mask(self, mask):
        """
        Post-process the action mask after all constraints have been applied.
        """

        batch_size = self.td_state.batch_size

        #   Depot must always be open for active agents regardless of other constraints
        depot_idx_exp = self.td_state['depot_idx'].unsqueeze(1).expand(*batch_size, self.num_agents, 1)  # [B, A, 1]
        active_agents = self.td_state['agents']['active_agents_mask']                    # [B, A]
        mask.scatter_(2, depot_idx_exp, active_agents.unsqueeze(-1))

        # Zero out inactive agents
        active_expanded = self.td_state['agents']['active_agents_mask'].unsqueeze(-1).expand(-1, -1, self.num_nodes)
        mask = mask & active_expanded

        # After done, close all services
        done = self.td_state['done']  # [B] or [B, 1]
        mask = mask & ~done.unsqueeze(-1).unsqueeze(-1)

        # Open depot for agent 0 only in done batch rows
        done_rows = done.squeeze(-1).nonzero(as_tuple=True)[0]
        if done_rows.numel() > 0:
            depot_idx = self.td_state['depot_idx'].squeeze(-1) if self.td_state['depot_idx'].dim() > 1 else self.td_state['depot_idx']
            mask[done_rows, 0, depot_idx[done_rows]] = True

        # force_visit: agents at depot with unvisited nodes must leave
        if self.force_visit:
            cur_node = self.td_state['agents']['cur_node_idx']          # [B, A]
            depot_idx = self.td_state['depot_idx']                      # [B, 1]
            at_depot = (cur_node == depot_idx)                          # [B, A]
            has_unvisited = self.td_state['nodes']['active_nodes_mask'][:, 1:].any(dim=-1)  # [B]
            must_leave = at_depot & has_unvisited.unsqueeze(-1)         # [B, A]
            depot_idx_expanded = depot_idx.unsqueeze(1).expand(*batch_size, self.num_agents, 1)  # [B, A, 1]
            depot_open = mask.gather(2, depot_idx_expanded) & ~must_leave.unsqueeze(-1)         # [B, A, 1]
            mask.scatter_(2, depot_idx_expanded, depot_open)

        return mask


    def _update_done(self, actions):
        """
        Update done state for all agents simultaneously.

        Args:
            actions(torch.Tensor): [B, A] tensor with all agents' moves.

        Returns:
            None.
        """
        self._update_done_all(actions)


    def _update_state(self, actions):
        """
        Update environment state for all agents simultaneously.

        Args:
            actions(torch.Tensor): [B, A] tensor with all agents' moves.

        Returns:
            None.
        """
        self._update_state_all(actions)


    def _update_solution(self, actions):

        """
        Update agents and actions in solution.

        Args:
            actions(torch.Tensor): Tensor with agents moves.

        Returns:
            None.
        """

        # update solution dic
        if 'actions' in self.td_state['solution'].keys():
            self.td_state['solution','actions'] = torch.concat( [self.td_state['solution','actions'], actions], dim=-1)
        else:
            self.td_state['solution','actions'] = actions

        if 'agents' in self.td_state['solution'].keys():
            self.td_state['solution','agents'] = torch.concat( [self.td_state['solution','agents'], self.td_state['cur_agent_idx']], dim=-1)
        else:
            self.td_state['solution','agents'] = self.td_state['cur_agent_idx']

    def _update_done_all(self, actions):
        """
        Update done state for all agents simultaneously.

        Args:
            actions(torch.Tensor): [B, A] tensor with all agents' moves.

        Returns:
            None.
        """
        former_done = self.td_state['done'].clone()

        depot_idx = self.td_state['depot_idx']           # [B, 1]
        went_to_depot = actions.eq(depot_idx)            # [B, A]

        # An agent becomes inactive when it moves to the depot
        self.td_state['agents']['active_agents_mask'] = (
            self.td_state['agents']['active_agents_mask'] & ~went_to_depot
        )

        self.td_state['done'] = (~self.td_state['agents']['active_agents_mask']).all(dim=-1)
        self.td_state['done'][former_done] = True

        # Depot stays active (scatter True); customer nodes become inactive (scatter False)
        self.td_state['nodes']['active_nodes_mask'].scatter_(1, actions, went_to_depot)

        self.td_state['is_last_step'] = self.td_state['done'].eq(~former_done)

    def _update_state_all(self, actions):
        """
        Update environment state for all agents simultaneously.

        Args:
            actions(torch.Tensor): [B, A] tensor with all agents' moves.

        Returns:
            None.
        """
        batch_size = self.td_state.batch_size

        # Current and next locations for every agent: [B, A, 2]
        cur_nodes = self.td_state['agents']['cur_node_idx']   # [B, A]
        loc = self.td_state['coords'].gather(
            1, cur_nodes.unsqueeze(-1).expand(*batch_size, self.num_agents, 2)
        )
        next_loc = self.td_state['coords'].gather(
            1, actions.unsqueeze(-1).expand(*batch_size, self.num_agents, 2)
        )

        ptime = self.td_state['agents']['cur_time'].clone()   # [B, A]
        distance2j = get_distance(loc, next_loc)              # [B, A]
        time2j = distance2j / self.td_state['speed']          # [B, A]
        if self.n_digits is not None:
            distance2j = torch.floor(self.n_digits * distance2j) / self.n_digits
            time2j = torch.floor(self.n_digits * time2j) / self.n_digits

        time_update = ptime + time2j   # [B, A]

        # Agents that are now done (went to depot in this step, already inactive)
        agents_done = ~self.td_state['agents']['active_agents_mask']   # [B, A]

        # Update current node and time for all agents
        self.td_state['agents']['cur_node_idx'] = actions
        self.td_state['agents']['cur_time'] = time_update

        # Update travel-time accumulators
        self.td_state['agents']['cur_travel_time'] = time2j
        self.td_state['agents']['cum_travel_time'] = self.td_state['agents']['cum_travel_time'] + time2j

        # Update load: subtract demand of visited node; zero out loads of done agents
        demands_taken = self.td_state['demands'].gather(1, actions)   # [B, A]
        new_load = self.td_state['agents']['cur_load'] - demands_taken
        new_load = torch.where(agents_done, torch.zeros_like(new_load), new_load)
        self.td_state['agents']['cur_load'] = new_load

        # Zero out demands of visited customer nodes
        self.td_state['nodes']['cur_demands'].scatter_(
            1, actions, torch.zeros_like(actions, dtype=torch.float)
        )

        # Mark visited nodes: visited_nodes[b, a, actions[b,a]] = True
        self.td_state['agents']['visited_nodes'].scatter_(
            2,
            actions.unsqueeze(-1),
            torch.ones(*batch_size, self.num_agents, 1, dtype=torch.bool, device=self.device),
        )

        # Increment step counter only for agents that are still active
        self.td_state['agents']['cur_step'] = torch.where(
            ~agents_done,
            self.td_state['agents']['cur_step'] + 1,
            self.td_state['agents']['cur_step'],
        )

        # If all agents are done, activate agent 0 to keep batch consistency
        self.td_state['agents']['active_agents_mask'][
            self.td_state['agents']['active_agents_mask'].sum(1).eq(0), 0
        ] = True

    def _update_solution_all(self, actions):
        """
        Update solution with all agents' actions for one simultaneous step.

        Args:
            actions(torch.Tensor): [B, A] tensor with all agents' moves.

        Returns:
            None.
        """
        batch_size = self.td_state.batch_size
        # Agent indices for this step: [B, A]
        agent_indices = (
            torch.arange(self.num_agents, device=self.device)
            .unsqueeze(0)
            .expand(*batch_size, -1)
        )

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

    def step(self, td: TensorDict) -> TensorDict:
        """
        Perform an environment step for active agent.

        Args:
            td(TensorDict): Environment tensor instance.

        Returns:
            td(TensorDict): Updated environment tensor instance.
        """


        actions = td["next_actions"]
        assert self.td_state['cur_agent']['action_mask'].gather(1, actions).all(), f"not feasible action"

        self._update_done(actions)
        done = self.td_state['done'].clone()
        is_last_step = self.td_state['is_last_step'].clone()

        # update env state
        self._update_state(actions)

        # update solution dic
        self._update_solution(actions)

        # get reward and penalty
        reward, penalty = self.reward_evaluator.get_reward(actions)

        self.env_nsteps += 1
        td.update(
            {
                "reward": reward,
                "penalty":penalty,
                "done": done,
                "is_last_step": is_last_step
            },
        )
        return td

    def step_observe(self, td: TensorDict,
                    obs_list: Optional[List[str]] = ['agents_action_mask']) -> TensorDict:

        """
        Perform an environment step for active agent.

        Args:
            td(TensorDict): Environment tensor instance.
            obs_list (Optional[List[str]]): List of observation keys to include. Defaults to ['agents_action_mask'].

        Returns:
            td(TensorDict): Updated environment tensor instance.
        """
        td = self.step(td)
        td = self.observe(td, obs_list=obs_list)
        return td

    def step_all(self, td: TensorDict) -> TensorDict:
        """
        Perform a simultaneous environment step for ALL agents.

        td["next_actions"] must be a [B, A] tensor with one action per agent.
        All agents act in parallel in a single call.

        Args:
            td(TensorDict): Environment tensor instance.  Must contain
                ``"next_actions"`` of shape ``[B, A]``.

        Returns:
            td(TensorDict): Updated environment tensor instance.
        """
        actions = td["next_actions"]   # [B, A]
        # Only check feasibility for active agents; inactive agents have all-False
        # masks (depot included) by design, but their action is a harmless no-op.
        active_agents = self.td_state['agents']['active_agents_mask']  # [B, A]
        if active_agents.any():
            assert (
                self.td_state['agents']['action_mask']
                .gather(2, actions.unsqueeze(-1))
                .squeeze(-1)[active_agents]
                .all()
            ), "Not all actions are feasible"

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
            }
        )
        return td

    def step_all_observe(self, td: TensorDict,
                         obs_list: Optional[List[str]] = ['agents_action_mask']) -> TensorDict:
        """
        Perform a simultaneous environment step for ALL agents, then observe.

        Args:
            td(TensorDict): Environment tensor instance.  Must contain
                ``"next_actions"`` of shape ``[B, A]``.
            obs_list (Optional[List[str]]): List of observation keys to include.
                Defaults to ``['agents_action_mask']``.

        Returns:
            td(TensorDict): Updated environment tensor instance with observations.
        """
        td = self.step_all(td)
        td = self.observe(td, obs_list=obs_list)
        return td

    def check_solution_validity(self):
        """
        Check if solution is valid according to CVRPTW constraints.

        Args:
            N/a.

        Returns:
            None. Raises AssertionError if invalid.
        """
        curr_node = torch.zeros(*self.batch_size, dtype=torch.int64, device=self.device)
        curr_time = torch.zeros(*self.batch_size, dtype=torch.float32, device=self.device)
        visited_nodes = torch.zeros(*self.batch_size, self.num_nodes, dtype=torch.int64, device=self.device)

        sorted_indices = torch.argsort(self.td_state['solution']['agents'], dim=-1, stable=True)
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

            # Time window constraints
            arrivej = curr_time + time

            # Mark node as visited
            fill = visited_nodes.gather(1, next_node.unsqueeze(-1))
            visited_nodes.scatter_(1, next_node.unsqueeze(-1), fill + 1)

            # Update time
            curr_time = arrivej
            curr_node = next_node
            curr_time[next_node == 0] = 0.0

        visited_nodes_exc_depot = visited_nodes[:, 1:]
        assert torch.all((visited_nodes_exc_depot == 0) | (visited_nodes_exc_depot == 1)), "Nodes were visited more than once!"
        eps = 1e-6
        # c1: capacity constraint
        used_cap = torch.zeros_like(self.td_state['demands'][:, 0])
        for ii in range(sorted_data.size(1)):
            #reset at depot
            used_cap = used_cap * (sorted_data[:, ii] != 0)
            used_cap += demand[:, ii]
            cap = self.td_state['capacity']
            # allow small numerical tolerance: either strictly <= or numerically close within atol=eps
            feasible_cap = (used_cap <= cap) | torch.isclose(used_cap, cap, atol=eps, rtol=0.0)
            assert torch.all(feasible_cap), "Agent exceeded vehicle capacity."
