.. _TOP-agent-agent-selector:

===============================
Agent Selector
===============================

Agent selector settings are defined in file ``env_agent_selector.py``.

RoundRobin
----------------------

Selects agents in a round-robin fashion (cyclic order).

.. autoclass:: maenvs4vrp.environments.top.env_agent_selector.RoundRobin
    :members:
    :special-members: __init__

RandomSelector
----------------------

Selects randomly between active agents.

.. autoclass:: maenvs4vrp.environments.top.env_agent_selector.RandomSelector
    :members:
    :special-members: __init__

AgentSelector
----------------------

Selects agents in order by index, skipping done agents.

.. autoclass:: maenvs4vrp.environments.top.env_agent_selector.AgentSelector
    :members:
    :special-members: __init__

SmallestTimeAgentSelector
---------------------------

Selects the active agent with the smallest current time.

.. autoclass:: maenvs4vrp.environments.top.env_agent_selector.SmallestTimeAgentSelector
    :members:
    :special-members: __init__
