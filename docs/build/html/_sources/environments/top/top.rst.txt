:hide-toc:

==========
TOP
==========

Team Orienteering Problem (TOP)

In a TOP instance, a set of $n$ nodes $\{ \mathrm{n}_i \}_{i=1}^n$ with their corresponding coordinates $x_i \in \mathbb{R}^2$ and a $n \times n$ symmetric matrix $T$ with travel time between each pair of locations are given. Every node $\mathrm{n}_i$ has a positive score or reward $r_i$ and a visit duration $d_i$. Without loss of generality, we can assume that $\mathrm{n}_1$ is the starting and ending location for every route. The objective is to find $m$ routes with the maximum possible sum of scores, without repeating visits, starting each route on or after a given time $t_{start}$ and ending before time $t_{end}$.

The objective is to maximize the total profit collected across all vehicle routes.

Here's everything about TOP environment:

.. toctree::
    :maxdepth: 1
    
    agent-reward/agent-reward
    agent-selector/agent-selector
    environment/environment
    generation/generation
    observations/observations
