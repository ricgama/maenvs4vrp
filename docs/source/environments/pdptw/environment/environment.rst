.. _PDPTW-environment:

===============
Environment
===============

PDPTW environment operations.

Environment settings are defined in file ``env.py``.

Environment
-------------

.. autoclass:: maenvs4vrp.environments.pdptw.env.Environment

.. autofunction:: maenvs4vrp.environments.pdptw.env.Environment.__init__

.. autofunction:: maenvs4vrp.environments.pdptw.env.Environment.observe

.. autofunction:: maenvs4vrp.environments.pdptw.env.Environment.sample_action

.. autofunction:: maenvs4vrp.environments.pdptw.env.Environment.reset

.. autofunction:: maenvs4vrp.environments.pdptw.env.Environment._update_feasibility

.. autofunction:: maenvs4vrp.environments.pdptw.env.Environment._update_done

.. autofunction:: maenvs4vrp.environments.pdptw.env.Environment._update_state

.. autofunction:: maenvs4vrp.environments.pdptw.env.Environment._update_cur_agent

.. autofunction:: maenvs4vrp.environments.pdptw.env.Environment._update_solution

.. autofunction:: maenvs4vrp.environments.pdptw.env.Environment.step