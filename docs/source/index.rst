===========
MAEnvs4VRP
===========

MAEnvs4VRP is a library made up of multi-agent environments for simulating classic vehicle routing problems.
The library provides:

* A flexible modular architecture.
* Design that allows easy customization.
* Incorporation of new routing problems.

In order to narrow the current gap between the test beds for algorithm benchmarking used in Reinforcement Learning and Operations Research (OR) communities, the library allows a straightforward integration of classical OR benchmark
instances.

If you want to have a grasp of the basics about the library, you can check our `quickstart notebook <content/start.html>`_.


.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: Getting Started

    content/install
    content/start
    content/test

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: MAEnvs4VRP Architecture

    core/modules

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: Environments
    
    environments/cvrpstw/cvrpstw
    environments/cvrptw/cvrptw
    environments/dsvrptw/dsvrptw
    environments/dvrptw/dvrptw
    environments/gmtdvrp/gmtdvrp
    environments/gmtvrp/gmtvrp
    environments/htop/htop
    environments/mtdvrp/mtdvrp
    environments/mdvrptw/mdvrptw
    environments/mtvrp/mtvrp
    environments/pcvrptw/pcvrptw
    environments/pdptw/pdptw
    environments/sdvrptw/sdvrptw
    environments/top/top
    environments/toptw/toptw

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: Utils

    utils/plotting

.. toctree::
    :maxdepth: 1
    :hidden:
    :titlesonly:
    :caption: Tutorials

    notebooks/1.0.0_quickstart_cvrptw.ipynb
    notebooks/2.0.0_maenvs4vrp_exploration_and_challenges.ipynb
    notebooks/3.0.0_multitask_environments.ipynb
    notebooks/4.0.0_maenvs4vrp_stochastic_environments.ipynb
    notebooks/5.0.0_PyVRP_cvrptw_solver.ipynb
