============
Installation
============

The library requires Python 3.11 or higher for installation, and it has been tested and confirmed stable with Python 3.13.5.

Quick Install
-------------

Install directly from PyPI:

.. code:: shell

    pip install maenvs4vrp

Development Install
-------------------

For a full development setup (notebooks, training scripts, and tests), isolate dependencies using a virtual environment.

With conda:

.. code:: shell

    conda create --name maenvs4vrp python=3.13.5
    conda activate maenvs4vrp

Then clone and install locally:

.. code:: shell

    git clone https://github.com/MAEnvs4VRP/maenvs4vrp.git && cd maenvs4vrp
    pip install -e .
