.. _MDVRPTW-generation-benchmark-generation:

===============================
Benchmark Instance Generation 
===============================

MDVRPTW environment instances are introduced in [Vid2013]_:

Benchmark instances generation settings are defined in file ``benchmark_instances_generator.py``.

Data files format
-------------------

.. code-block:: text

    ************************
    * MDVRPTW instances    *
    ************************

    The first line contains the following information:

        type m n t d

    where

        type = 8 (MDPVRP)

        m = number of vehicles

        n = number of customers

        t = number of days

        d = number of depots


    The next t lines contain, for each day/depot, the following information:

            D Q

    where

        D = maximum duration of a route

        Q = maximum load of a vehicle


    The next lines contain, for each depot and for each customer, the following information:

        i x y d q f a list e l

    where

        i = customer/depot number

        x = x coordinate

        y = y coordinate

        d = service duration

        q = demand

        f = frequency of visit

        a = number of possible visit combinations

        list = list of all possible visit combinations

            e = beginning of time window (earliest time for start of service),
                if any

            l = end of time window (latest time for start of service), if any
    
                Each visit combination is coded with the decimal equivalent of
                the corresponding binary bit string. For example, in a 5-day
                period, the code 10 which is equivalent to the bit string 01010
                means that a customer is visited on days 2 and 4. (Days are
                numbered from left to right.)


BenchmarkInstanceGenerator
---------------------------------

.. autoclass:: maenvs4vrp.environments.mdvrptw.benchmark_instances_generator.BenchmarkInstanceGenerator

.. autofunction:: maenvs4vrp.environments.mdvrptw.benchmark_instances_generator.BenchmarkInstanceGenerator.__init__

.. autofunction:: maenvs4vrp.environments.mdvrptw.benchmark_instances_generator.BenchmarkInstanceGenerator.get_list_of_benchmark_instances

.. autofunction:: maenvs4vrp.environments.mdvrptw.benchmark_instances_generator.BenchmarkInstanceGenerator.read_instance_data

.. autofunction:: maenvs4vrp.environments.mdvrptw.benchmark_instances_generator.BenchmarkInstanceGenerator.parse_instance_data

.. autofunction:: maenvs4vrp.environments.mdvrptw.benchmark_instances_generator.BenchmarkInstanceGenerator.get_instance 

.. autofunction:: maenvs4vrp.environments.mdvrptw.benchmark_instances_generator.BenchmarkInstanceGenerator.load_set_of_instances

.. autofunction:: maenvs4vrp.environments.mdvrptw.benchmark_instances_generator.BenchmarkInstanceGenerator.sample_first_n_services

.. autofunction:: maenvs4vrp.environments.mdvrptw.benchmark_instances_generator.BenchmarkInstanceGenerator.random_sample_instance

.. autofunction:: maenvs4vrp.environments.mdvrptw.benchmark_instances_generator.BenchmarkInstanceGenerator.sample_name_from_set

.. autofunction:: maenvs4vrp.environments.mdvrptw.benchmark_instances_generator.BenchmarkInstanceGenerator.sample_instance

**References**

.. [Vid2013] M, T. Vidal, T. G. Crainic, M. Gendreau, and C. Prins, A hybrid genetic algorithm with adaptive diversity management for a large class of vehicle routing problems with time-windows, Comput. Oper. Res., vol. 40, no. 1, pp. 475-489, 2013.