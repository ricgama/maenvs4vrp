.. _DVRPTW-agent-agent-reward:

===============================
Agent Rewards
===============================

The reward is 0 in all steps except the last. At the end of the episode, the reward is the negative of the sum of the distances of the routes traveled by all agents, minus the sum of the penalties for each service not performed. The penalty for a not performed service is 2 times the distance from the depot to that service.

Agent reward settings are defined in file ``env_agent_reward.py``.

Dense Reward
----------------

At every step, the reward is the negative distance traveled by the agent. At the end of the episode, a penalty is given equaling $10$ times the negative distance from the depot to the not attended services.

.. autoclass:: maenvs4vrp.environments.dvrptw.env_agent_reward.DenseReward
    :members:
    :special-members: __init__

Sparse Reward
----------------

The reward is 0 in all steps except the last. At the end of the episode, the reward is the negative of the sum of the distances of the routes traveled by all agents minus the sum of the penalties for each service not performed.
The penalty for a not-performed service is $10$ times the distance from the depot to that service.

.. autoclass:: maenvs4vrp.environments.dvrptw.env_agent_reward.SparseReward
    :members:
    :special-members: __init__