.. _PCVRPTW-observations:

===============
Observations
===============

PCVRPTW observations operations.

Observations settings are defined in file ``observations.py``.

Observations
------------------

.. autoclass:: maenvs4vrp.environments.pcvrptw.observations.Observations

.. autofunction:: maenvs4vrp.environments.pcvrptw.observations.Observations.__init__

.. autofunction:: maenvs4vrp.environments.pcvrptw.observations.Observations.set_env

Nodes static features
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: maenvs4vrp.environments.pcvrptw.observations.Observations.get_nodes_static_feat_dim

.. autofunction:: maenvs4vrp.environments.pcvrptw.observations.Observations.get_nodes_dynamic_feat_dim

.. autofunction:: maenvs4vrp.environments.pcvrptw.observations.Observations.get_nodes_feat_dim

.. autofunction:: maenvs4vrp.environments.pcvrptw.observations.Observations.get_agent_feat_dim

.. autofunction:: maenvs4vrp.environments.pcvrptw.observations.Observations.get_other_agents_feat_dim

.. autofunction:: maenvs4vrp.environments.pcvrptw.observations.Observations.get_global_feat_dim

.. autofunction:: maenvs4vrp.environments.pcvrptw.observations.Observations.get_feat_x_coordinate

.. autofunction:: maenvs4vrp.environments.pcvrptw.observations.Observations.get_feat_y_coordinate

.. autofunction:: maenvs4vrp.environments.pcvrptw.observations.Observations.get_feat_x_coordinate_min_max

.. autofunction:: maenvs4vrp.environments.pcvrptw.observations.Observations.get_feat_y_coordinate_min_max

.. autofunction:: maenvs4vrp.environments.pcvrptw.observations.Observations.get_feat_tw_low

.. autofunction:: maenvs4vrp.environments.pcvrptw.observations.Observations.get_feat_tw_high

.. autofunction:: maenvs4vrp.environments.pcvrptw.observations.Observations.get_feat_demand

.. autofunction:: maenvs4vrp.environments.pcvrptw.observations.Observations.get_feat_service_time

.. autofunction:: maenvs4vrp.environments.pcvrptw.observations.Observations.get_feat_tw_high_minus_tw_low_div_max_dur

.. autofunction:: maenvs4vrp.environments.pcvrptw.observations.Observations.get_feat_is_depot

Nodes dynamic features
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: maenvs4vrp.environments.pcvrptw.observations.Observations.get_feat_time2open_div_end_time

.. autofunction:: maenvs4vrp.environments.pcvrptw.observations.Observations.get_feat_time2close_div_end_time

.. autofunction:: maenvs4vrp.environments.pcvrptw.observations.Observations.get_feat_arrive2node_div_end_time

.. autofunction:: maenvs4vrp.environments.pcvrptw.observations.Observations.get_feat_time2open_after_step_div_end_time

.. autofunction:: maenvs4vrp.environments.pcvrptw.observations.Observations.get_feat_time2close_after_step_div_end_time

.. autofunction:: maenvs4vrp.environments.pcvrptw.observations.Observations.get_feat_time2end_after_step_div_end_time

.. autofunction:: maenvs4vrp.environments.pcvrptw.observations.Observations.get_feat_fract_time_after_step_div_end_time

.. autofunction:: maenvs4vrp.environments.pcvrptw.observations.Observations.get_feat_reachable_frac_agents

Current agent features
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: maenvs4vrp.environments.pcvrptw.observations.Observations.get_feat_agent_x_coordinate

.. autofunction:: maenvs4vrp.environments.pcvrptw.observations.Observations.get_feat_agent_y_coordinate

.. autofunction:: maenvs4vrp.environments.pcvrptw.observations.Observations.get_feat_agent_x_coordinate_min_max

.. autofunction:: maenvs4vrp.environments.pcvrptw.observations.Observations.get_feat_agent_y_coordinate_min_max

.. autofunction:: maenvs4vrp.environments.pcvrptw.observations.Observations.get_feat_agent_frac_current_time

.. autofunction:: maenvs4vrp.environments.pcvrptw.observations.Observations.get_feat_agent_frac_current_load

.. autofunction:: maenvs4vrp.environments.pcvrptw.observations.Observations.get_feat_agent_arrivedepot_div_end_time

.. autofunction:: maenvs4vrp.environments.pcvrptw.observations.Observations.get_feat_agent_frac_feasible_nodes

.. autofunction:: maenvs4vrp.environments.pcvrptw.observations.Observations.get_feat_agents_dist2depot_div_end_time

Other agents features
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: maenvs4vrp.environments.pcvrptw.observations.Observations.get_feat_agents_x_coordinate

.. autofunction:: maenvs4vrp.environments.pcvrptw.observations.Observations.get_feat_agents_y_coordinate

.. autofunction:: maenvs4vrp.environments.pcvrptw.observations.Observations.get_feat_agents_x_coordinate_min_max

.. autofunction:: maenvs4vrp.environments.pcvrptw.observations.Observations.get_feat_agents_y_coordinate_min_max

.. autofunction:: maenvs4vrp.environments.pcvrptw.observations.Observations.get_feat_agents_frac_current_time

.. autofunction:: maenvs4vrp.environments.pcvrptw.observations.Observations.get_feat_agents_frac_current_load

.. autofunction:: maenvs4vrp.environments.pcvrptw.observations.Observations.get_feat_agents_frac_feasible_nodes

.. autofunction:: maenvs4vrp.environments.pcvrptw.observations.Observations.get_feat_agents_dist2agent_div_end_time

.. autofunction:: maenvs4vrp.environments.pcvrptw.observations.Observations.get_feat_agents_time_delta2agent_div_max_dur

.. autofunction:: maenvs4vrp.environments.pcvrptw.observations.Observations.get_feat_agents_was_last

Global features
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: maenvs4vrp.environments.pcvrptw.observations.Observations.get_feat_global_frac_done_agents

.. autofunction:: maenvs4vrp.environments.pcvrptw.observations.Observations.get_feat_global_frac_demands

.. autofunction:: maenvs4vrp.environments.pcvrptw.observations.Observations.get_feat_global_frac_fleet_load_capacity

Computing features
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: maenvs4vrp.environments.pcvrptw.observations.Observations.compute_static_features

.. autofunction:: maenvs4vrp.environments.pcvrptw.observations.Observations.compute_dynamic_features

.. autofunction:: maenvs4vrp.environments.pcvrptw.observations.Observations.compute_agent_features

.. autofunction:: maenvs4vrp.environments.pcvrptw.observations.Observations.compute_agents_features

.. autofunction:: maenvs4vrp.environments.pcvrptw.observations.Observations.compute_global_features

.. autofunction:: maenvs4vrp.environments.pcvrptw.observations.Observations.get_observations

Internal methods
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: maenvs4vrp.environments.pcvrptw.observations.Observations._concat_features

.. autofunction:: maenvs4vrp.environments.pcvrptw.observations.Observations._normalize_feature

.. autofunction:: maenvs4vrp.environments.pcvrptw.observations.Observations._min_max_normalization

.. autofunction:: maenvs4vrp.environments.pcvrptw.observations.Observations._min_max_normalization2d

.. autofunction:: maenvs4vrp.environments.pcvrptw.observations.Observations._standardize