============
Installation
============

For a clean setup, isolate library dependencies using a virtual environment. The library requires Python 3.11 or higher for installation, and it has been tested and confirmed stable with Python 3.13.5.
To create an isolated environment with conda:

.. code:: shell

    conda create --name maenvs4vrp python=3.13.5
    conda activate maenvs4vrp

To install MAENVS4VRP locally on your machine:

.. code:: shell

    git clone https://github.com/ricgama/maenvs4vrp.git && cd maenvs4vrp
    pip install -e .
