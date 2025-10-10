=====================
MAEnvs4VRP Quickstart
=====================

Welcome to the MAEnvs4VRP Quickstart guide!  
This tutorial demonstrates the **basic usage** of the library through the ``CVRPTW`` environment example.

If you prefer interactive examples, you can also explore the corresponding Jupyter notebooks in <https://maenvs4vrp.readthedocs.io/en/latest/notebooks/1.0.0_quickstart_cvrptw.html>`_.

------------------
Basic Usage
------------------

Let's walk through a simple example using the ``CVRPTW`` environment to understand the main components of MAEnvs4VRP.

Import Libraries
================

Start by importing all the core modules needed for creating and interacting with an environment.  
Here, we exclude benchmarking and toy instance generators for simplicity.

.. code-block:: python

    from maenvs4vrp.environments.cvrptw.env import Environment
    from maenvs4vrp.environments.cvrptw.env_agent_selector import AgentSelector
    from maenvs4vrp.environments.cvrptw.observations import Observations
    from maenvs4vrp.environments.cvrptw.instances_generator import InstanceGenerator
    from maenvs4vrp.environments.cvrptw.env_agent_reward import DenseReward

    %load_ext autoreload
    %autoreload 2

Generate Instances and Create the Environment
=============================================

Now, let's generate instances and assemble the environment.  
We first create each individual component — instance generator, observations, agent selector, and reward evaluator — and then combine them into a complete environment.

.. code-block:: python

    gen = InstanceGenerator(batch_size=8)
    obs = Observations()
    sel = AgentSelector()
    rew = DenseReward()

    env = Environment(
        instance_generator_object=gen,
        obs_builder_object=obs,
        agent_selector_object=sel,
        reward_evaluator=rew,
        seed=0
    )

Reset the Environment
=====================

Before simulation, the environment must be reset.  
This initializes the problem with the specified number of agents and nodes.

.. code-block:: python

    td = env.reset(batch_size=8, num_agents=4, num_nodes=16)

Run Simulation Steps
====================

Once the environment is initialized, agents can begin performing actions.  
The simulation proceeds until all agents have completed their tasks and returned to the depot.

.. code-block:: python

    while not td["done"].all():  
        td = env.sample_action(td)  # This is where your policy interacts with the environment
        td = env.step(td)

------------------
Continue Learning
------------------

This Quickstart provides a foundation for understanding how MAEnvs4VRP operates.  
You can find more hands-on examples and advanced topics in the following notebooks:

.. list-table::
   :widths: 20 30 50
   :header-rows: 1

   * - Notebook
     - Description
     - Colab
   * - `01: Quickstart <https://maenvs4vrp.readthedocs.io/en/latest/notebooks/1.0.0_quickstart_cvrptw.html>`_
     - Learning MAEnvs4VRP basic usage.
     - |colab-quickstart|
   * - `02: MAEnvs4VRP Library <https://maenvs4vrp.readthedocs.io/en/latest/notebooks/2.0.0_maenvs4vrp_exploration_and_challenges.html>`_
     - Exploring MAEnvs4VRP functionality and challenges.
     - |colab-challenges|
   * - `03: Multi-Tasking Environments <https://maenvs4vrp.readthedocs.io/en/latest/notebooks/3.0.0_multitask_environments.html>`_
     - Understanding multi-tasking behavior across environments.
     - |colab-multitask|
   * - `04: Stochastic Environments <https://maenvs4vrp.readthedocs.io/en/latest/notebooks/4.0.0_maenvs4vrp_stochastic_environments.html>`_
     - Extending deterministic environments into stochastic versions.
     - |colab-stochastic|
   * - `05: PyVRP <https://maenvs4vrp.readthedocs.io/en/latest/notebooks/5.0.0_PyVRP_cvrptw_solver.html>`_
     - Integrating PyVRP to solve MAEnvs4VRP instances.
     - |colab-PyVRP|

.. |colab-quickstart| image:: https://colab.research.google.com/assets/colab-badge.svg
   :alt: Google Colab Badge
   :target: https://colab.research.google.com/github/ricgama/maenvs4vrp/blob/master/maenvs4vrp/notebooks/1.0.0_quickstart_cvrptw.ipynb
.. |colab-challenges| image:: https://colab.research.google.com/assets/colab-badge.svg
   :alt: Google Colab Badge
   :target: https://colab.research.google.com/github/ricgama/maenvs4vrp/blob/master/maenvs4vrp/notebooks/2.0.0_maenvs4vrp_exploration_and_challenges.ipynb
.. |colab-multitask| image:: https://colab.research.google.com/assets/colab-badge.svg
   :alt: Google Colab Badge
   :target: https://colab.research.google.com/github/ricgama/maenvs4vrp/blob/master/maenvs4vrp/notebooks/3.0.0_multitask_environments.ipynb
.. |colab-stochastic| image:: https://colab.research.google.com/assets/colab-badge.svg
   :alt: Google Colab Badge
   :target: https://colab.research.google.com/github/ricgama/maenvs4vrp/blob/master/maenvs4vrp/notebooks/4.0.0_maenvs4vrp_stochastic_environments.ipynb
.. |colab-PyVRP| image:: https://colab.research.google.com/assets/colab-badge.svg
   :alt: Google Colab Badge
   :target: https://colab.research.google.com/github/ricgama/maenvs4vrp/blob/master/maenvs4vrp/notebooks/5.0.0_PyVRP_cvrptw_solver.ipynb
