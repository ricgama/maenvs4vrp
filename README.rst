**********
MAENVS4VRP
**********

Multi Agent Environments for Vehicle Routing Problems

MAENVS4VRP is a library made up of multi-agent environments for simulating classic vehicle routing problems.

`Documentation <https://maenvs4vrp.readthedocs.io/en/latest/>`_ | `Install <#id13>`_ | `Quickstart Notebook <https://maenvs4vrp.readthedocs.io/en/latest/content/start.html>`_ | `Train Your Model <#training>`_ | `Paper <https://arxiv.org/abs/2411.14411>`_

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: Google Colab Badge
    :target: https://colab.research.google.com/github/ricgama/maenvs4vrp/blob/master/maenvs4vrp/notebooks/1.0.0-quickstart-cvrptw.ipynb

Environments
============

.. list-table:: List of Vehicle Routing Environments :
   :widths: 25 5 5
   :header-rows: 1

   * - Evironment
     - Source
     - Description
   * - CVRPSTW (Capacitated Vehicle Routing Problem with Soft Time Windows)
     - `Code <https://github.com/ricgama/maenvs4vrp/tree/master/maenvs4vrp/environments/cvrpstw>`_
     - `Docs <https://maenvs4vrp.readthedocs.io/en/latest/environments/cvrpstw/cvrpstw.html>`_
   * - CVRPTW (Capacitated Vehicle Routing Problem with Time Windows)
     - `Code <https://github.com/ricgama/maenvs4vrp/tree/master/maenvs4vrp/environments/cvrptw>`_
     - `Docs <https://maenvs4vrp.readthedocs.io/en/latest/environments/cvrptw/cvrptw.html>`_
   * - MDVRPTW (Multi-trip Vehicle Routing Problem with Time Windows)
     - `Code <https://github.com/ricgama/maenvs4vrp/tree/master/maenvs4vrp/environments/mdvrptw>`_
     - `Docs <https://maenvs4vrp.readthedocs.io/en/latest/environments/mdvrptw/mdvrptw.html>`_
   * - PCVRPTW (Prize Collecting Vehicle Routing Problem with Time Windows)
     - `Code <https://github.com/ricgama/maenvs4vrp/tree/master/maenvs4vrp/environments/pcvrptw>`_
     - `Docs <https://maenvs4vrp.readthedocs.io/en/latest/environments/pcvrptw/pcvrptw.html>`_
   * - PDPTW (Pickup and Delivery Problem with Time Windows)
     - `Code <https://github.com/ricgama/maenvs4vrp/tree/master/maenvs4vrp/environments/pdptw>`_
     - `Docs <https://maenvs4vrp.readthedocs.io/en/latest/environments/pdptw/pdptw.html>`_
   * - SDVRPTW (Split Delivery Vehicle Routing Problem with Time Windows)
     - `Code <https://github.com/ricgama/maenvs4vrp/tree/master/maenvs4vrp/environments/sdvrptw>`_
     - `Docs <https://maenvs4vrp.readthedocs.io/en/latest/environments/sdvrptw/sdvrptw.html>`_
   * - TOPTW (Team Orienteering Problem with Time Windows)
     - `Code <https://github.com/ricgama/maenvs4vrp/tree/master/maenvs4vrp/environments/toptw>`_
     - `Docs <https://maenvs4vrp.readthedocs.io/en/latest/environments/toptw/toptw.html>`_

Install
==========

If you want to install MAENVS4VRP locally on your machine:

.. code:: shell

    git clone https://github.com/ricgama/maenvs4vrp.git && cd maenvs4vrp
    pip install -e .

Testing
=============

You can run tests in testing directory ``/tests/unit/environments``.

.. code-block:: python

  pytest seed_test.py

.. code-block:: python

  pytest reset_seed.py

Training
=============

You can train models in learning directory ``maenvs4vrp/learning``.

.. code-block:: python

    python maenvs4vrp/learning/mardam/train_mardam.py --vrp_env toptw --num_agents 5 --num_nodes 51  --val_set servs_50_agents_5 --selection stime

.. code-block:: python

    python maenvs4vrp/learning/madyam/train_madyam.py --vrp_env toptw --num_agents 5 --num_nodes 51  --val_set servs_50_agents_5 --selection stime

Directory Tree Structure
===========================

.. code:: text

    ├───maenvs4vrp
    │   ├───core
    │   ├───environments
    │   │   ├───cvrpstw
    │   │   ├───cvrptw
    │   │   ├───mdvrptw
    │   │   ├───pcvrptw
    │   │   ├───pdptw
    │   │   ├───sdvrptw
    │   │   ├───toptw
    │   ├───learning
    │   │   ├───madyam
    │   │   ├───mardam
    │   ├───notebooks
    │   ├───utils
    ├───tests
    │   ├───unit
    │   │   ├───environments

Citation
===============

If you want to include MAENVS4VRP in your publications, you can use:

.. code-block:: bibtex

    @article{gama2024maenvs4vrp,
      title={Multi-Agent Environments for Vehicle Routing Problems},
      author={Ricardo Gama and Daniel Fuertes and Carlos R. del-Blanco and Hugo L. Fernandes},
      year={2024},
      journal={arXiv preprint arXiv:2411.14411},
      note={\url{https://github.com/ricgama/maenvs4vrp}}
      url={https://arxiv.org/abs/2411.14411},
    }
