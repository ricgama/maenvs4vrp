:hide-toc:

==========
PDPTW
==========

Pickup and Delivery with Time Windows (PDPTW)

In the pickup and delivery problem with time windows (see [Dum1997]), a fleet of vehicles with uniform capacity has to collect and deliver items to satisfy pairs of customers, respecting their visiting time window. Concretely, an instance consists of a fleet of $V$ homogeneous vehicles with maximum capacity $Q$. A set of $2n+1$ nodes $\{ \mathrm{n}_i \}_{i=1}^{2n+1}$, a depot and a set $\{1, \ldots, n\}$ of pickup services and their corresponding $\{n+1, \ldots, 2n\}$ delivery services, with coordinates $x_i \in \mathbb{R}^2$, a $2n+1 \times 2n+1$ symmetric matrix $T$ with travel time between each pair of locations. Each customer with a pick-up service has a quantity $d_i$ to be delivered to a particular customer $\mathrm{n}_{n+i}$. In addition, each node $\mathrm{n}_i$ is specified with a visiting time window $[o_i, c_i]$ with opening time and closing time, and service time $s_i$.

Here's everything about PDPTW environment:

.. toctree::
    :maxdepth: 1
    
    agent-reward/agent-reward
    agent-selector/agent-selector
    environment/environment
    generation/generation
    observations/observations

References
--------------

.. [Dum1997] DUMAS, Yvan ; DESROSIERS, Jacques ; SOUMIS, François: The pickup and delivery problem with time windows. In: European Journal of Operational Research 54 (1991)