=================================
MAENVS4VRP Quickstart Notebook
=================================

Basic Usage
=============

Let's explore the library using the CVRPTW environment as an example.

Import libraries
^^^^^^^^^^^^^^^^^^^^

Include all libraries from the environment, except benchmarking and toy generation.

.. code-block:: python

    from maenvs4vrp.environments.cvrptw.env import Environment
    from maenvs4vrp.environments.cvrptw.env_agent_selector import AgentSelector
    from maenvs4vrp.environments.cvrptw.observations import Observations
    from maenvs4vrp.environments.cvrptw.instances_generator import InstanceGenerator
    from maenvs4vrp.environments.cvrptw.env_agent_reward import DenseReward
    %load_ext autoreload
    %autoreload 2

Generate instances and create the environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create each instance to generator, observations, agent selector and rewards. Afterwards, you can use all these instances to create your environment.

.. code-block:: python

    gen = InstanceGenerator(batch_size = 8)
    obs = Observations()
    sel = AgentSelector()
    rew = DenseReward()

    env = Environment(instance_generator_object=gen,  
                    obs_builder_object=obs,
                    agent_selector_object=sel,
                    reward_evaluator=rew,
                    seed=0)

Reset environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before simulating the problem, you must reset the environment.

.. code-block:: python

    td = env.reset(batch_size = 8, num_agents=4, num_nodes=16)

Compute agent steps until all agents return to depot
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Until all agents are done with their tasks and back to depot, the prgram will iteratively perform an action and move to the next agent.

.. code-block:: python

    while not td["done"].all():  
        td = env.sample_action(td) # this is where we insert our policy
        td = env.step(td)