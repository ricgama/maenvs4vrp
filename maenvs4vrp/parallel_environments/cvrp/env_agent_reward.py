import torch
from tensordict import TensorDict
from maenvs4vrp.core.env_agent_reward import RewardFn

from typing import Optional, List
from maenvs4vrp.utils.ops import get_distance


class DenseReward(RewardFn):
    """
    CVRP dense reward class.
    """

    def __init__(self):
        """
        Constructor.

        Args:
            n/a.

        Returns:
            None.
        """
        self.env = None
        self.pending_penalty = -10

    def set_env(self, env):

        """
        Set environment.

        Args:
            env(AECEnv): Environment.

        Returns:
            None.
        """

        self.env = env

    def get_reward(self, actions):
        """
        Get reward and penalty.

        Args:
            action(torch.Tensor): [B, A] tensor with all agents' moves.

        Returns:
            reward(torch.Tensor): Per-agent reward of shape [B, A].
            penalty(torch.Tensor): Per-agent penalty of shape [B, A].
        """

        # Each agent's reward = negative travel time it spent on this step
        reward = -self.env.td_state['agents']['cur_travel_time'].clone()   # [B, A]
        penalty = torch.zeros(*self.env.batch_size, self.env.num_agents, dtype=torch.float, device=self.env.device)  # [B, A]

        # compute penalty if env has unvisited nodes at the last step
        is_last_step = self.env.td_state['is_last_step']   # [B]

        dist_depot2nodes = torch.pairwise_distance(self.env.td_state['depot_loc'], self.env.td_state['coords'], eps=0, keepdim=False)
        time_depot2nodes = dist_depot2nodes / self.env.td_state['speed']

        if self.env.n_digits is not None:
            dist_depot2nodes = torch.floor(self.env.n_digits * dist_depot2nodes) / self.env.n_digits
            time_depot2nodes = torch.floor(self.env.n_digits * time_depot2nodes) / self.env.n_digits

        # Global penalty scalar per batch element: [B]
        unvisited_penalty = (
            self.pending_penalty
            * (time_depot2nodes * self.env.td_state['nodes']['active_nodes_mask'])
            .sum(-1).float()
        )
        # Assign equally to all agents in the last-step rows: [n_done] -> [n_done, A]
        penalty[is_last_step] = (
            unvisited_penalty[is_last_step].unsqueeze(-1).expand(-1, self.env.num_agents)
        )

        return reward, penalty



class SparseReward(RewardFn):
    """
    CVRP sparse reward class.
    """

    def __init__(self):
        """
        Constructor.

        Args:
            n/a.

        Returns:
            None.
        """
        self.env = None
        self.pending_penalty = -10

    def set_env(self, env):

        """
        Set environment.

        Args:
            env(Environment): Environment.

        Returns:
            None.
        """

        self.env = env

    def get_reward(self, actions):
        """
        Get reward and penalty.

        Args:
            actions(torch.Tensor): [B, A] tensor with all agents' moves.

        Returns:
            reward(torch.Tensor): Reward.
            penalty(torch.Tensor): Penalty.
        """

        # Each agent's reward and penalty: [B, A]
        reward = torch.zeros(*self.env.batch_size, self.env.num_agents, dtype=torch.float, device=self.env.device)
        penalty = torch.zeros(*self.env.batch_size, self.env.num_agents, dtype=torch.float, device=self.env.device)

        # compute penalty if env has unvisited nodes
        is_last_step = self.env.td_state['is_last_step']   # [B]

        dist_depot2nodes = torch.pairwise_distance(self.env.td_state['depot_loc'], self.env.td_state['coords'], eps=0, keepdim=False)
        time_depot2nodes = dist_depot2nodes / self.env.td_state['speed']

        if self.env.n_digits is not None:
            dist_depot2nodes = torch.floor(self.env.n_digits * dist_depot2nodes) / self.env.n_digits
            time_depot2nodes = torch.floor(self.env.n_digits * time_depot2nodes) / self.env.n_digits

        # Each agent's final reward = its own negative cumulative travel time: [B, A]
        final_reward = -self.env.td_state['agents']['cum_travel_time']   # [B, A]

        # Global penalty per batch element: [B]
        unvisited_penalty = (
            self.pending_penalty
            * (time_depot2nodes * self.env.td_state['nodes']['active_nodes_mask'])
            .sum(-1).float()
        )
        # Assign equally to all agents in the last-step rows: [n_done] -> [n_done, A]
        penalty[is_last_step] = (
            unvisited_penalty[is_last_step].unsqueeze(-1).expand(-1, self.env.num_agents)
        )

        reward[is_last_step] = final_reward[is_last_step]
        return reward, penalty
