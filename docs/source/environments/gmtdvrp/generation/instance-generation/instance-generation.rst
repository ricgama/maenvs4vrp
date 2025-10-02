.. _GMTDVRP-generation-instance-generation:

======================
Instance Generation 
======================

Instance generation follows the paper [Li21]_:

#. The depot and services’ $(x, y)$ locations are sampled uniformly from $[0, 1]^2$;
#. Each service’s demand $d_i$ is sampled uniformly from ${1, 2, ..., 9}$, and each vehicle has capacity $C = 50$;
#. For the time window constraint, we set the time window for the depot as $[b_0, e_0] = [0, 3]$, and the service time at each $i$ to be $s_i = 0.2$. We further set the time window for city node $i$ by: (a) sampling the time window center $ci ∼ U([b_0 + t_{0,i}, e_0 − t_{i,0} − s_i])$, where $t_{0,i} = t_{i,0}$ is the travel time, equaling the Euclidean distance, from the depot to node $i$; (b) sampling the time window half-width $h_i$ uniformly at random from $[s_i/2, e_0/3] = [0.1, 1]$; (c) setting the time window for $i$ as $[max(b_0, c_i − h_i), min(e_0, c_i + h_i)]$.

Instances generation settings are defined in file ``instances_generator.py``.

InstanceGenerator 
--------------------

.. autoclass:: maenvs4vrp.environments.gmtdvrp.instances_generator.InstanceGenerator
    :members:
    :special-members: __init__

**References**

.. [Li21] Li, Z. Yan, C. Wu, “Learning to delegate for large-scale vehicle routing”, Thirty-Fifth Conference on Neural Information Processing Systems, 2021;