:hide-toc:

==========
CVRPTW
==========

Capacitated Vehicle Routing Problem with Time Windows (CVRPTW)

On the CRPTW we have a fleet of vehicles to serve a set of customers with known demands. Each customer can only be served once by a single vehicle. The service must be carried out within a defined time window and respecting the vehicle's capacities (see [Sol87]_). Concretely, an instance consists of a fleet of $V$ homogeneous vehicles, a set of $n$ nodes $\{ \mathrm{n}_i \}_{i=1}^n$, a depot and a set of $n-1$ services, with their corresponding coordinates $x_i \in \mathbb{R}^2$, a $n \times n$ symmetric matrix $T$ with travel time between each pair of locations, a quantity $d_i$ that specifies the demand for some resource by each customer $\mathrm{n}_i$, and the maximum quantity, $Q$, of the resource that a vehicle can carry. In addition, each node $\mathrm{n}_i$ is specified with a visiting time window $[o_i , c_i]$ with opening time and closing time, and service time $s_i$. Unless otherwise stated, time window constraints are considered hard constraints, i.e. a vehicle is allowed to arrive at a customer location before its opening time, but it must wait to make the delivery. If it arrives after its closing time, it's not allowed to serve the respective customer.

The objective is to minimize the total combined traveled time of all vehicle routes.

Here's everything about CVRPTW environment:

.. toctree::
    :maxdepth: 1
    
    agent-reward/agent-reward
    agent-selector/agent-selector
    environment/environment
    generation/generation
    observations/observations

**References**

.. [Sol87] M, M. Solomon, "Algorithms for the vehicle routing and scheduling problems with time window constraints", Operations Research, 35(2):254–265, 1987;