.. _PDPTW-generation-benchmark-generation:

===============================
Benchmark Instance Generation 
===============================

`Li&Lim <https://www.sintef.no/projectweb/top/pdptw/li-lim-benchmark/>`_ instances are included to be used with PDPTW environment.

Benchmark instances generation settings are defined in file ``benchmark_instances_generator.py``.

Data files format
--------------------

.. code-block:: text

    ************************
    * PDPTW instances      *
    ************************

    Below is an explanation of the format of the instance definitions (text files). Note that tabulator is used as field separator rather than spaces. 

    NUMBER OF VEHICLES  VEHICLE CAPACITY SPEED(not used)
    K   Q   S



    TASK NO. X Y DEMAND EARLIEST PICKUP/DELIVERY TIME LATEST PICKUP/DELIVERY TIME SERVICE TIME PICKUP(index to sibling) DELIVERY(index to sibling)

    0   x0  y0  q0  e0  l0  s0  p0 d0

    …   …   …   …   …  …   …   …   …

    

    Task 0 specifies the coordinates of the depot.  For pickup tasks, the PICKUP index is 0, whereas the DELIVERY sibling gives the index of the corresponding delivery task. For delivery tasks, the PICKUP index gives the index of the corresponding pickup task. The value of travel time is equal to the value of distance.

BenchmarkInstanceGenerator
---------------------------------

.. autoclass:: maenvs4vrp.environments.pdptw.benchmark_instances_generator.BenchmarkInstanceGenerator

.. autofunction:: maenvs4vrp.environments.pdptw.benchmark_instances_generator.BenchmarkInstanceGenerator.__init__

.. autofunction:: maenvs4vrp.environments.pdptw.benchmark_instances_generator.BenchmarkInstanceGenerator.get_list_of_benchmark_instances

.. autofunction:: maenvs4vrp.environments.pdptw.benchmark_instances_generator.BenchmarkInstanceGenerator.read_instance_data

.. autofunction:: maenvs4vrp.environments.pdptw.benchmark_instances_generator.BenchmarkInstanceGenerator.parse_instance_data

.. autofunction:: maenvs4vrp.environments.pdptw.benchmark_instances_generator.BenchmarkInstanceGenerator.get_instance 

.. autofunction:: maenvs4vrp.environments.pdptw.benchmark_instances_generator.BenchmarkInstanceGenerator.load_set_of_instances

.. autofunction:: maenvs4vrp.environments.pdptw.benchmark_instances_generator.BenchmarkInstanceGenerator.sample_first_n_services

.. autofunction:: maenvs4vrp.environments.pdptw.benchmark_instances_generator.BenchmarkInstanceGenerator.random_sample_instance

.. autofunction:: maenvs4vrp.environments.pdptw.benchmark_instances_generator.BenchmarkInstanceGenerator.sample_name_from_set

.. autofunction:: maenvs4vrp.environments.pdptw.benchmark_instances_generator.BenchmarkInstanceGenerator.sample_instance