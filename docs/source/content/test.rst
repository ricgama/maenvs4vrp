=======================
Testing Environment 
=======================

We have some tests you can use to make sure the library is running well. We have two types of testing you should run before simulating problems. You can find them in the directory ``tests/unit/environments``

Seed Test
^^^^^^^^^^^^

Here, all environments are tested according to:

* Seeds: Checks if same seeds generate same output and if different seeds generate different outputs.
* Instance generator: Tests seed equivalence for benchmarking instance generators and normal instance generators.

To run the test you can use:

.. code-block:: python

    pytest seed_test.py

Reset Test
^^^^^^^^^^^^^^

Reset test assures that environments behave correctly, based on different settings. We have 3 types of tests here:

* Reset tests: Checks if benchmarking and normal instances can be reset without error.
* Observation tests: Checks if environment presents observations correctly.
* Agent iteration tests: Checks if iterations to select agents are done corrently in different types types of agent selectors and generator instances.

To run the test you can use:

.. code-block:: python

    pytest reset_test.py