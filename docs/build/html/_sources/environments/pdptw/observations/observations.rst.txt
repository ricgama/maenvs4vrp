.. _PDPTW-observations:

===============
Observations
===============

PDPTW observations operations.

Observations settings are defined in file ``observations.py``.

Observations
------------------

.. autoclass:: maenvs4vrp.environments.pdptw.observations.Observations
    :members: __init__, set_env

Nodes static features
^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: maenvs4vrp.environments.pdptw.observations.Observations.get_feat_x_coordinate

.. automethod:: maenvs4vrp.environments.pdptw.observations.Observations.get_feat_y_coordinate

.. automethod:: maenvs4vrp.environments.pdptw.observations.Observations.get_feat_x_coordinate_min_max

.. automethod:: maenvs4vrp.environments.pdptw.observations.Observations.get_feat_y_coordinate_min_max

.. automethod:: maenvs4vrp.environments.pdptw.observations.Observations.get_feat_tw_low

.. automethod:: maenvs4vrp.environments.pdptw.observations.Observations.get_feat_tw_high

.. automethod:: maenvs4vrp.environments.pdptw.observations.Observations.get_feat_demand

.. automethod:: maenvs4vrp.environments.pdptw.observations.Observations.get_feat_service_time

.. automethod:: maenvs4vrp.environments.pdptw.observations.Observations.get_feat_tw_high_minus_tw_low_div_max_dur

.. automethod:: maenvs4vrp.environments.pdptw.observations.Observations.get_feat_is_depot

.. automethod:: maenvs4vrp.environments.pdptw.observations.Observations.get_feat_is_pickup

.. automethod:: maenvs4vrp.environments.pdptw.observations.Observations.get_feat_is_delivery

Nodes dynamic features
^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: maenvs4vrp.environments.pdptw.observations.Observations.get_feat_time2open_div_end_time

.. automethod:: maenvs4vrp.environments.pdptw.observations.Observations.get_feat_time2close_div_end_time

.. automethod:: maenvs4vrp.environments.pdptw.observations.Observations.get_feat_arrive2node_div_end_time

.. automethod:: maenvs4vrp.environments.pdptw.observations.Observations.get_feat_time2open_after_step_div_end_time

.. automethod:: maenvs4vrp.environments.pdptw.observations.Observations.get_feat_time2close_after_step_div_end_time

.. automethod:: maenvs4vrp.environments.pdptw.observations.Observations.get_feat_time2end_after_step_div_end_time

.. automethod:: maenvs4vrp.environments.pdptw.observations.Observations.get_feat_fract_time_after_step_div_end_time

.. automethod:: maenvs4vrp.environments.pdptw.observations.Observations.get_feat_reachable_frac_agents

.. automethod:: maenvs4vrp.environments.pdptw.observations.Observations.get_feat_is_pending

Current agent features
^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: maenvs4vrp.environments.pdptw.observations.Observations.get_feat_agent_x_coordinate

.. automethod:: maenvs4vrp.environments.pdptw.observations.Observations.get_feat_agent_y_coordinate

.. automethod:: maenvs4vrp.environments.pdptw.observations.Observations.get_feat_agent_x_coordinate_min_max

.. automethod:: maenvs4vrp.environments.pdptw.observations.Observations.get_feat_agent_y_coordinate_min_max

.. automethod:: maenvs4vrp.environments.pdptw.observations.Observations.get_feat_agent_frac_current_time

.. automethod:: maenvs4vrp.environments.pdptw.observations.Observations.get_feat_agent_frac_current_load

.. automethod:: maenvs4vrp.environments.pdptw.observations.Observations.get_feat_agent_arrivedepot_div_end_time

.. automethod:: maenvs4vrp.environments.pdptw.observations.Observations.get_feat_agent_frac_feasible_nodes

Other agents features
^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: maenvs4vrp.environments.pdptw.observations.Observations.get_feat_agents_x_coordinate

.. automethod:: maenvs4vrp.environments.pdptw.observations.Observations.get_feat_agents_y_coordinate

.. automethod:: maenvs4vrp.environments.pdptw.observations.Observations.get_feat_agents_x_coordinate_min_max

.. automethod:: maenvs4vrp.environments.pdptw.observations.Observations.get_feat_agents_y_coordinate_min_max

.. automethod:: maenvs4vrp.environments.pdptw.observations.Observations.get_feat_agents_frac_current_time

.. automethod:: maenvs4vrp.environments.pdptw.observations.Observations.get_feat_agents_frac_current_load

.. automethod:: maenvs4vrp.environments.pdptw.observations.Observations.get_feat_agents_frac_feasible_nodes

.. automethod:: maenvs4vrp.environments.pdptw.observations.Observations.get_feat_agents_dist2agent_div_end_time

.. automethod:: maenvs4vrp.environments.pdptw.observations.Observations.get_feat_agents_dist2depot_div_end_time

.. automethod:: maenvs4vrp.environments.pdptw.observations.Observations.get_feat_agents_time_delta2agent_div_max_dur

.. automethod:: maenvs4vrp.environments.pdptw.observations.Observations.get_feat_agents_was_last

Global features
^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: maenvs4vrp.environments.pdptw.observations.Observations.get_feat_global_frac_done_agents

.. automethod:: maenvs4vrp.environments.pdptw.observations.Observations.get_feat_global_frac_demands

.. automethod:: maenvs4vrp.environments.pdptw.observations.Observations.get_feat_global_frac_fleet_load_capacity

Computing features
^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: maenvs4vrp.environments.pdptw.observations.Observations.compute_static_features

.. automethod:: maenvs4vrp.environments.pdptw.observations.Observations.compute_dynamic_features

.. automethod:: maenvs4vrp.environments.pdptw.observations.Observations.compute_agent_features

.. automethod:: maenvs4vrp.environments.pdptw.observations.Observations.compute_agents_features

.. automethod:: maenvs4vrp.environments.pdptw.observations.Observations.compute_global_features

.. automethod:: maenvs4vrp.environments.pdptw.observations.Observations.get_observations

Internal methods
^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: maenvs4vrp.environments.pdptw.observations.Observations._concat_features

.. automethod:: maenvs4vrp.environments.pdptw.observations.Observations._normalize_feature

.. automethod:: maenvs4vrp.environments.pdptw.observations.Observations._min_max_normalization

.. automethod:: maenvs4vrp.environments.pdptw.observations.Observations._min_max_normalization2d

.. automethod:: maenvs4vrp.environments.pdptw.observations.Observations._standardize