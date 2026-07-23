================
Observations
================

This class is responsible for the computation of the observation features that will be available to the active agent while it's interacting with the environment. 

There are five types of observations possible:

* `nodes\_static` - Nodes/locations intrinsic features (e.g. location, time window width, demands, profits, etc); 

* `nodes\_dynamic` - Nodes/locations step dependent features. Usually, these observations are computed in relation to the active agent  (e.g. fraction of time used by the agent after node visit, time left for location opening, time left for location closing);

* `agent` - Active agent-related features (e.g. fraction of visited nodes, fraction of feasible nodes, fraction of current load, fraction of time available);

* `other\_agents` - Features regarding all agents still active in the environment (e.g. location, fraction of time available, fraction of used capacity, distance to the active agent, time difference to the active agent);

* `global` - Environment's global state features (e.g. fraction of completed services, fraction of used agents, fraction of demands satisfied, fraction of profits collected);

ObservationBuilder
---------------------

.. autoclass:: maenvs4vrp.core.env_observation_builder.ObservationBuilder

.. autofunction:: maenvs4vrp.core.env_observation_builder.ObservationBuilder.__init__

.. autofunction:: maenvs4vrp.core.env_observation_builder.ObservationBuilder.set_env

.. autofunction:: maenvs4vrp.core.env_observation_builder.ObservationBuilder.get_static_feat_dim

.. autofunction:: maenvs4vrp.core.env_observation_builder.ObservationBuilder.get_dynamic_feat_dim

.. autofunction:: maenvs4vrp.core.env_observation_builder.ObservationBuilder.get_nodes_feat_dim

.. autofunction:: maenvs4vrp.core.env_observation_builder.ObservationBuilder.get_agent_feat_dim

.. autofunction:: maenvs4vrp.core.env_observation_builder.ObservationBuilder.get_other_agents_feat_dim

.. autofunction:: maenvs4vrp.core.env_observation_builder.ObservationBuilder.get_global_feat_dim

.. autofunction:: maenvs4vrp.core.env_observation_builder.ObservationBuilder.compute_static_features

.. autofunction:: maenvs4vrp.core.env_observation_builder.ObservationBuilder.compute_dynamic_features

.. autofunction:: maenvs4vrp.core.env_observation_builder.ObservationBuilder.compute_agent_features

.. autofunction:: maenvs4vrp.core.env_observation_builder.ObservationBuilder.compute_agents_features

.. autofunction:: maenvs4vrp.core.env_observation_builder.ObservationBuilder.compute_global_features

.. autofunction:: maenvs4vrp.core.env_observation_builder.ObservationBuilder.get_observations