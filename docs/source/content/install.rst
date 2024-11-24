=====================
Installation
=====================

You can use any software you like to create an isolated python environment, so that all dependencies are consistent with the project. Make sure you `clone the repository <https://github.com/ricgama/maenvs4vrp_dev>`_ first and enter its directory in your command line.

Virtualenv
-----------

You can check how to install virtualenv `here <https://virtualenv.pypa.io/en/latest/installation.html>`_.

Create environment
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    virtualenv maenvs4vrp

Activate environment
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    maenvs4vrp\Scripts\activate #windows

.. code-block:: bash

    maenvs4vrp/bin/activate #macos or linux

Install project
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    pip install -e .


Anaconda
-----------

You can install anaconda `here <https://www.anaconda.com/download/>`_.

Create environment
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    conda create --name maenvs4vrp python=3.11

Activate environment
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    conda activate maenvs4vrp

Install project
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    pip install -e .