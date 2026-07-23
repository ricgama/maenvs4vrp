.. _TOP-observations:

===============
Observations
===============

TOP observations.

Observations settings are defined in file ``observations.py``.

Observations
------------------

.. autoclass:: maenvs4vrp.environments.top.observations.Observations
    :members: __init__, set_env

Nodes static features
^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: maenvs4vrp.environments.top.observations.Observations.get_feat_x_coordinate

.. automethod:: maenvs4vrp.environments.top.observations.Observations.get_feat_y_coordinate

.. automethod:: maenvs4vrp.environments.top.observations.Observations.get_feat_x_coordinate_min_max

.. automethod:: maenvs4vrp.environments.top.observations.Observations.get_feat_y_coordinate_min_max

.. automethod:: maenvs4vrp.environments.top.observations.Observations.get_feat_profits

.. automethod:: maenvs4vrp.environments.top.observations.Observations.get_feat_service_time

.. automethod:: maenvs4vrp.environments.top.observations.Observations.get_feat_is_depot

Nodes dynamic features
^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: maenvs4vrp.environments.top.observations.Observations.get_feat_arrive2node_div_end_time

.. automethod:: maenvs4vrp.environments.top.observations.Observations.get_feat_time2end_after_step_div_end_time

.. automethod:: maenvs4vrp.environments.top.observations.Observations.get_feat_fract_time_after_step_div_end_time

.. automethod:: maenvs4vrp.environments.top.observations.Observations.get_feat_reachable_frac_agents

Current agent features
^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: maenvs4vrp.environments.top.observations.Observations.get_feat_agent_x_coordinate

.. automethod:: maenvs4vrp.environments.top.observations.Observations.get_feat_agent_y_coordinate

.. automethod:: maenvs4vrp.environments.top.observations.Observations.get_feat_agent_x_coordinate_min_max

.. automethod:: maenvs4vrp.environments.top.observations.Observations.get_feat_agent_y_coordinate_min_max

.. automethod:: maenvs4vrp.environments.top.observations.Observations.get_feat_agent_frac_current_time

.. automethod:: maenvs4vrp.environments.top.observations.Observations.get_feat_agent_frac_current_profit

.. automethod:: maenvs4vrp.environments.top.observations.Observations.get_feat_agent_arrivedepot_div_end_time

.. automethod:: maenvs4vrp.environments.top.observations.Observations.get_feat_agent_frac_feasible_nodes

Other agents features
^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: maenvs4vrp.environments.top.observations.Observations.get_feat_other_agents_x_coordinate

.. automethod:: maenvs4vrp.environments.top.observations.Observations.get_feat_other_agents_y_coordinate

.. automethod:: maenvs4vrp.environments.top.observations.Observations.get_feat_other_agents_frac_current_time

.. automethod:: maenvs4vrp.environments.top.observations.Observations.get_feat_other_agents_time_delta2agent_div_max_dur

.. automethod:: maenvs4vrp.environments.top.observations.Observations.get_feat_other_agents_frac_time_left

.. automethod:: maenvs4vrp.environments.top.observations.Observations.get_feat_other_agents_frac_current_profit

.. automethod:: maenvs4vrp.environments.top.observations.Observations.get_feat_other_agents_was_last

All agents features
^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: maenvs4vrp.environments.top.observations.Observations.get_feat_all_agents_x_coordinate

.. automethod:: maenvs4vrp.environments.top.observations.Observations.get_feat_all_agents_y_coordinate

.. automethod:: maenvs4vrp.environments.top.observations.Observations.get_feat_all_agents_frac_current_time

.. automethod:: maenvs4vrp.environments.top.observations.Observations.get_feat_all_agents_frac_current_profit

Global features
^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: maenvs4vrp.environments.top.observations.Observations.get_feat_global_frac_done_agents

.. automethod:: maenvs4vrp.environments.top.observations.Observations.get_feat_global_frac_profits

.. automethod:: maenvs4vrp.environments.top.observations.Observations.get_feat_global_frac_colect_profits
