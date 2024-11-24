.. _CVRPTW-generation-benchmark-generation:

===============================
Benchmark Instance Generation 
===============================

Solomon and Homberger instances are included to be used with CVRPTW environment.

Benchmark instances generation settings are defined in file ``benchmark_instances_generator.py``.

Data files format
--------------------

.. code-block:: text

    ************************
    * CVRPTW instances     *
    ************************

    Data files format:

    For these problem, data files format is as follows:


    The first line contains the following data, number of paths, all vertices can be visited and number of vertices.

    From the 3th line, each line contains the integer data associated to each customer, starting with the depot:

    vertex number
    x coordinate
    y coordinate
    service duration or visiting time
    profit of the location
    Opening of time window
    Closing of time window
    Gavalas

    For these problem, data files format is as follows:

    The first line contains the following data, number of tours, start day,number of vertices.

    The second line associties with start/end point:

    vertex number = 0
    x coordinate
    y coordinate
    visiting duration = 0
    score of the location = 0
    Opening of time window
    Closing of time window
    The remaining lines contain the data of each point. For each point, the line contains the following data:

    vertex number
    x coordinate
    y coordinate
    visiting duration
    score of the location
    Opening hour of the day i, i = 0,1,2,3,4,5,6
    Closing hour of the day i, i = 0,1,2,3,4,5,6

BenchmarkInstanceGenerator
---------------------------------

.. autoclass:: maenvs4vrp.environments.cvrptw.benchmark_instances_generator.BenchmarkInstanceGenerator

.. autofunction:: maenvs4vrp.environments.cvrptw.benchmark_instances_generator.BenchmarkInstanceGenerator.__init__

.. autofunction:: maenvs4vrp.environments.cvrptw.benchmark_instances_generator.BenchmarkInstanceGenerator.get_list_of_benchmark_instances

.. autofunction:: maenvs4vrp.environments.cvrptw.benchmark_instances_generator.BenchmarkInstanceGenerator.read_instance_data

.. autofunction:: maenvs4vrp.environments.cvrptw.benchmark_instances_generator.BenchmarkInstanceGenerator.parse_instance_data

.. autofunction:: maenvs4vrp.environments.cvrptw.benchmark_instances_generator.BenchmarkInstanceGenerator.get_instance 

.. autofunction:: maenvs4vrp.environments.cvrptw.benchmark_instances_generator.BenchmarkInstanceGenerator.load_set_of_instances

.. autofunction:: maenvs4vrp.environments.cvrptw.benchmark_instances_generator.BenchmarkInstanceGenerator.sample_first_n_services

.. autofunction:: maenvs4vrp.environments.cvrptw.benchmark_instances_generator.BenchmarkInstanceGenerator.random_sample_instance

.. autofunction:: maenvs4vrp.environments.cvrptw.benchmark_instances_generator.BenchmarkInstanceGenerator.sample_name_from_set

.. autofunction:: maenvs4vrp.environments.cvrptw.benchmark_instances_generator.BenchmarkInstanceGenerator.sample_instance