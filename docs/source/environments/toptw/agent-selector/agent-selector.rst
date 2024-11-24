.. _TOPTW-agent-agent-selector:

===============================
Agent Selector
===============================

Agents can be selected under different criteria: The one which has lesser value, randomly or the one which is more avaliable.

Agent selector settings are defined in file ``env_agent_selector.py``.

AgentSelector
----------------------

Selects the same agent until it returns to the depot. Afterward, it selects the next active agent and repeats the process until all agents are done.

.. autoclass:: maenvs4vrp.environments.toptw.env_agent_selector.AgentSelector

.. autofunction:: maenvs4vrp.environments.toptw.env_agent_selector.AgentSelector.__init__

.. autofunction:: maenvs4vrp.environments.toptw.env_agent_selector.AgentSelector.set_env

.. autofunction:: maenvs4vrp.environments.toptw.env_agent_selector.AgentSelector._next_agent

RandomSelector
----------------------

Selects randomly between active agents. 

.. autoclass:: maenvs4vrp.environments.toptw.env_agent_selector.RandomSelector

.. autofunction:: maenvs4vrp.environments.toptw.env_agent_selector.RandomSelector.__init__

.. autofunction:: maenvs4vrp.environments.toptw.env_agent_selector.RandomSelector.set_env

.. autofunction:: maenvs4vrp.environments.toptw.env_agent_selector.RandomSelector._next_agent

SmallestTimeAgentSelector
---------------------------

Selects the agent with the smallest cumulated time since departure from the depot.

.. autoclass:: maenvs4vrp.environments.toptw.env_agent_selector.SmallestTimeAgentSelector

.. autofunction:: maenvs4vrp.environments.toptw.env_agent_selector.SmallestTimeAgentSelector.__init__

.. autofunction:: maenvs4vrp.environments.toptw.env_agent_selector.SmallestTimeAgentSelector.set_env

.. autofunction:: maenvs4vrp.environments.toptw.env_agent_selector.SmallestTimeAgentSelector._next_agent