.. _PCVRPTW-agent-agent-reward:

===============================
Agent Rewards
===============================

The reward is 0 in all steps except the last. At the end of the episode, the reward is the negative of the sum of the distances of the routes traveled by all agents, minus the sum of the penalties for each service not performed. The penalty for a not performed service is 2 times the distance from the depot to that service.

Agent reward settings are defined in file ``env_agent_reward.py``.

Dense Reward
----------------

At every step, the reward is the negative distance traveled plus the client prize collected by the agent.

.. autoclass:: maenvs4vrp.environments.pcvrptw.env_agent_reward.DenseReward

.. autofunction:: maenvs4vrp.environments.pcvrptw.env_agent_reward.DenseReward.__init__

.. autofunction:: maenvs4vrp.environments.pcvrptw.env_agent_reward.DenseReward.set_env

.. autofunction:: maenvs4vrp.environments.pcvrptw.env_agent_reward.DenseReward.get_reward

Sparse Reward
----------------

The reward is 0 in all steps except the last. At the end of the episode, the reward is the negative of the sum of the distances of the routes traveled plus the sum of all agents collected prizes.

.. autoclass:: maenvs4vrp.environments.pcvrptw.env_agent_reward.SparseReward

.. autofunction:: maenvs4vrp.environments.pcvrptw.env_agent_reward.SparseReward.__init__

.. autofunction:: maenvs4vrp.environments.pcvrptw.env_agent_reward.SparseReward.set_env

.. autofunction:: maenvs4vrp.environments.pcvrptw.env_agent_reward.SparseReward.get_reward