.. _CVRP-observations:

===============
Observations
===============

CVRP observations.

Observations settings are defined in file ``observations.py``.

Observations
------------------

.. autoclass:: maenvs4vrp.environments.cvrp.observations.Observations
    :members: __init__, set_env

Nodes static features
^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.get_feat_x_coordinate

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.get_feat_y_coordinate

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.get_feat_x_coordinate_min_max

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.get_feat_y_coordinate_min_max

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.get_feat_demand

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.get_feat_service_time

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.get_feat_is_depot

Edges static features
^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.get_feat_distance_matrix

Nodes dynamic features
^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.get_feat_reachable_frac_agents

Current agent features
^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.get_feat_agent_x_coordinate

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.get_feat_agent_y_coordinate

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.get_feat_agent_x_coordinate_min_max

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.get_feat_agent_y_coordinate_min_max

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.get_feat_agent_remaining_capacity

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.get_feat_agent_frac_feasible_nodes

Other agents features
^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.get_feat_other_agents_x_coordinate

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.get_feat_other_agents_y_coordinate

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.get_feat_other_agents_x_coordinate_min_max

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.get_feat_agents_y_coordinate_min_max

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.get_feat_other_agents_frac_current_load

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.get_feat_other_agents_remaining_capacity

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.get_feat_agents_frac_feasible_nodes

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.get_feat_other_agents_was_last

All agents features
^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.get_feat_all_agents_x_coordinate

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.get_feat_all_agents_y_coordinate

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.get_feat_all_agents_cur_time

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.get_feat_all_agents_remaining_capacity

Global features
^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.get_feat_global_frac_done_agents

.. automethod:: maenvs4vrp.environments.cvrp.observations.Observations.get_feat_global_frac_fleet_load_capacity
