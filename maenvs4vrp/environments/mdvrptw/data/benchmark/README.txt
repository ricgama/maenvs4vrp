MDVRPTW, PVRPTW, and SDVRPTW instances, introduced in

[1] T. Vidal, T. G. Crainic, M. Gendreau, and C. Prins, 
A hybrid genetic algorithm with adaptive diversity management for a large class of vehicle routing problems with time-windows,
Comput. Oper. Res., vol. 40, no. 1, pp. 475-489, 2013.

These instances are in a format similar to
Cordeau, J.-F., Gendreau, M., & Laporte, G. (1997). 
A tabu search heuristic for periodic and multi-depot vehicle routing problems. Networks, 30(2), 105-119.
(c.f. : http://neumann.hec.ca/chairedistributique/data/README.TXT)

The format of data and solution files is as follows:

A) DATA FILES

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
