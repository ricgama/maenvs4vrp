.. _CVRP-agent-agent-reward:

===============================
Agent Rewards
===============================

Agent reward settings are defined in file ``env_agent_reward.py``.

Dense Reward
----------------

At every step, the reward is the negative distance traveled by the agent. At the end of the episode, a penalty is given for each unvisited customer equaling $10$ times the distance from the depot to that customer.

.. autoclass:: maenvs4vrp.environments.cvrp.env_agent_reward.DenseReward
    :members:
    :special-members: __init__

Dense Reward V
----------------

Variant of the dense reward. At every step, the reward is the negative distance traveled by the agent. At the end of the episode, a penalty based on the number of unvisited nodes and active agent steps is applied.

.. autoclass:: maenvs4vrp.environments.cvrp.env_agent_reward.DenseRewardV
    :members:
    :special-members: __init__

Sparse Reward
----------------

The reward is 0 in all steps except the last. At the end of the episode, the reward is the negative of the sum of the distances of the routes traveled by all agents, minus the sum of the penalties for each service not performed. The penalty for a not-performed service is $10$ times the distance from the depot to that service.

.. autoclass:: maenvs4vrp.environments.cvrp.env_agent_reward.SparseReward
    :members:
    :special-members: __init__
