===========
MAENVS4VRP
===========

MAENVS4VRP is a library made up of multi-agent environments for simulating classic vehicle routing problems.
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
    :caption: MAENVS4VRP Architecture

    core/modules

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: Environments
    
    environments/cvrpstw/cvrpstw
    environments/cvrptw/cvrptw
    environments/htop/htop
    environments/mdvrptw/mdvrptw
    environments/pcvrptw/pcvrptw
    environments/pdptw/pdptw
    environments/sdvrptw/sdvrptw
    environments/top/top
    environments/toptw/toptw
    
.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: Tutorials

    notebooks/instance_generator.ipynb
    notebooks/observations.ipynb
    notebooks/agent_iteration.ipynb