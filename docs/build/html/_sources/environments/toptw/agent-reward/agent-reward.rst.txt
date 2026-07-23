.. _TOPTW-agent-agent-reward:

===============================
Agent Rewards
===============================

The reward is 0 in all steps except the last. At the end of the episode, the reward is the negative of the sum of the distances of the routes traveled by all agents, minus the sum of the penalties for each service not performed. The penalty for a not performed service is 2 times the distance from the depot to that service.

Agent reward settings are defined in file ``env_agent_reward.py``.

Dense Reward
----------------

At every step, the reward is the profit collected by the agent.

.. autoclass:: maenvs4vrp.environments.toptw.env_agent_reward.DenseReward
    :members:
    :special-members: __init__

Sparse Reward
----------------

The reward is 0 in all steps except the last. At the end of the episode, the reward is the sum of all agent's collected profits.

.. autoclass:: maenvs4vrp.environments.toptw.env_agent_reward.SparseReward
    :members:
    :special-members: __init__