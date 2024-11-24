.. _CVRPSTW-agent-agent-reward:

===============================
Agent Rewards
===============================

The reward is 0 in all steps except the last. At the end of the episode, the reward is the negative of the sum of the distances of the routes traveled by all agents, minus the sum of the penalties for each service not performed. The penalty for a not performed service is 2 times the distance from the depot to that service.

Agent reward settings are defined in file ``env_agent_reward.py``.

Dense Reward
----------------

At every step, the reward is the negative distance traveled by the agent plus the penalty for early and late time window arrival. At the end of the episode, a penalty is given equaling $10$ times the negative distance from the depot to the not attended services.

.. autoclass:: maenvs4vrp.environments.cvrpstw.env_agent_reward.DenseReward

.. autofunction:: maenvs4vrp.environments.cvrpstw.env_agent_reward.DenseReward.__init__

.. autofunction:: maenvs4vrp.environments.cvrpstw.env_agent_reward.DenseReward.set_env

.. autofunction:: maenvs4vrp.environments.cvrpstw.env_agent_reward.DenseReward.get_reward

Sparse Reward
----------------

The reward is 0 in all steps except the last. At the end of the episode, the reward is the negative of the sum of the distances of the routes traveled by all agents minus the sum of the penalties for each service not performed and time window violation.
The penalty for a not-performed service is $10$ times the distance from the depot to that service.

.. autoclass:: maenvs4vrp.environments.cvrpstw.env_agent_reward.SparseReward

.. autofunction:: maenvs4vrp.environments.cvrpstw.env_agent_reward.SparseReward.__init__

.. autofunction:: maenvs4vrp.environments.cvrpstw.env_agent_reward.SparseReward.set_env

.. autofunction:: maenvs4vrp.environments.cvrpstw.env_agent_reward.SparseReward.get_reward