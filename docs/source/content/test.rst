=======================
Testing Environment 
=======================

We have some tests you can use to make sure the library is running well. We have two types of testing you should run before simulating problems. You can find them in the directory ``tests/unit/environments``

Check Solution Test
^^^^^^^^^^^^^^^^^^^^^^

Script created to run tests on Multi-Tasking environments. It tests solutions validity on:

* Random instances: Different variants across batches based on random attribute sampling.
* All variants: Instances created on every variant available.
* Different agent selectors: All agent selectors are tested: AgentSelector, RandomSelector and SmallestTimeAgentSelector.
* Different configurations: Combinations of different numbers of agents and nodes.

To run the test, you can use:

.. code-block:: python

    pytest check_solution_test.py

Reset Test
^^^^^^^^^^^^^^

Reset test assures that environments behave correctly, based on different settings. We have 3 types of tests here:

* Reset tests: Checks if benchmarking and normal instances can be reset without error.
* Observation tests: Checks if environment presents observations correctly.
* Agent iteration tests: Checks if iterations to select agents are done corrently in different types types of agent selectors and generator instances.

To run the test, you can use:

.. code-block:: python

    pytest reset_test.py

Reset Benchmarking Test
^^^^^^^^^^^^^^^^^^^^^^^^^^

Reset benchmarking tests.

To run the test, you can use:

.. code-block:: python

    pytest reset_bench_test.py

Seed Test
^^^^^^^^^^^^

Here, all environments are tested according to:

* Seeds: Checks if same seeds generate same output and if different seeds generate different outputs.
* Instance generator: Tests seed equivalence for benchmarking instance generators and normal instance generators.

To run the test, you can use:

.. code-block:: python

    pytest seed_test.py

Seed Benchmarking Test
^^^^^^^^^^^^^^^^^^^^^^^^^^

Seed benchmarking tests.

To run the test, you can use:

.. code-block:: python

    pytest seed_bench_test.py