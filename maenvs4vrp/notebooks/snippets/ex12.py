class DenseReward(DenseReward):
    """Reward class.
    """

    def get_reward(self, action):
        """
        Get reward and penalty.

        Args:
            action(torch.Tensor): Tensor with agent moves.

        Returns:
            reward(torch.Tensor): Reward.
            penalty(torch.Tensor): Penalty.
        """

        reward = -self.env.td_state['cur_agent']['cur_ttime'].clone()
        penalty = self.env.td_state['cur_agent']['cur_penalty'].clone()

        # compute penalty if env has unvisited nodes 
        is_last_step = self.env.td_state['is_last_step']
        
        depot2nodes = torch.pairwise_distance(self.env.td_state['depot_loc'], self.env.td_state['coords'], eps=0, keepdim = False)
        if self.env.n_digits is not None:
            depot2nodes = torch.floor(self.env.n_digits * depot2nodes) / self.env.n_digits
        penalty[is_last_step] = self.pending_penalty * ((depot2nodes * self.env.td_state['nodes']['active_nodes_mask']).sum(-1, keepdim = True).float()[is_last_step])

        return reward, penalty