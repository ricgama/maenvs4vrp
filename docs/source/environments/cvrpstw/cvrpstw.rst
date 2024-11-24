:hide-toc:

===============
CVRPSTW
===============

Capacitated Vehicle Routing Problem with Soft Time Windows (CVRPSTW)

In this variation of the CVRPTW, time window constraints are relaxed and can be violated at a penalty cost (see, for example, usually linear proportional to the interval between opening/closing times and vehicle arrival. Although the penalty function can be defined in several ways, we consider the formulation studied in [Fil2010]_). Concretely, the time window violation cannot exceed $P_{max}$, and consequently, for each customer, we can enlarge its time window to $[o_i - P_{max}, c_i + P_{max}] = [o^s_i , c^s_i]$ outside which the service cannot be performed. When a vehicle arrives at a customer at time $t_i \in [o^s_i , c^s_i]$, it can have an early arrival penalty cost of $p_e \max (o_i-t_i,0)$ and a late arrival penalty cost of $p_l \max (t_i-c_i,0)$.

Furthermore, the vehicle's maximum waiting time at any customer, $W_{max}$, is imposed. That is, the vehicles can only arrive at each customer after $o_i - P_{max} - W_{max}$, so that its waiting time doesn't exceed $W_{max}$.

Here's everything about CVRPSTW environment:

.. toctree::
    :maxdepth: 1
    
    agent-reward/agent-reward
    agent-selector/agent-selector
    environment/environment
    generation/generation
    observations/observations

References
--------------
.. [Fil2010] FIGLIOZZI, Miguel A.: An iterative route construction and improvement algorithm for the vehicle routing problem with soft time windows. In: Transportation Research Part C: Emerging Technologies 18 (2010), Nr. 5, S. 668–679