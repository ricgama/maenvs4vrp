.. _CVRPTW-agent-agent-selector:

===============================
Agent Selector
===============================

Agents can be selected under different criteria: The one which has lesser value, randomly or the one which is more avaliable.

Agent selector settings are defined in file ``env_agent_selector.py``.

AgentSelector
----------------------

Selects the same agent until it returns to the depot. Afterward, it selects the next active agent and repeats the process until all agents are done.

.. autoclass:: maenvs4vrp.environments.cvrptw.env_agent_selector.AgentSelector
    :members:
    :special-members: __init__

RandomSelector
----------------------

Selects randomly between active agents. 

.. autoclass:: maenvs4vrp.environments.cvrptw.env_agent_selector.RandomSelector
    :members:
    :special-members: __init__

SmallestTimeAgentSelector
---------------------------

Selects the agent with the smallest cumulated time since departure from the depot.

.. autoclass:: maenvs4vrp.environments.cvrptw.env_agent_selector.SmallestTimeAgentSelector
    :members:
    :special-members: __init__