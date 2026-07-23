.. _TOP-agent-agent-reward:

===============================
Agent Rewards
===============================

Agent reward settings are defined in file ``env_agent_reward.py``.

Dense Reward
----------------

At every step, the reward is the profit collected by the agent at that step.

.. autoclass:: maenvs4vrp.environments.top.env_agent_reward.DenseReward
    :members:
    :special-members: __init__

Sparse Reward
----------------

The reward is 0 in all steps except the last. At the end of the episode, the reward is the total cumulative profit collected by all agents.

.. autoclass:: maenvs4vrp.environments.top.env_agent_reward.SparseReward
    :members:
    :special-members: __init__
