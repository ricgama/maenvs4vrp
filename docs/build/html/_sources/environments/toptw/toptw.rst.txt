:hide-toc:

==========
TOPTW
==========

Team Orienteering Problem with Time Windows (TOPTW)

In a TOPTW instance, a set of $n$ nodes, $\{ \mathrm{n}_i \}_{i=1}^n$ with their corresponding coordinates $x_i \in \mathbb{R}^2$ and a $n \times n$ symmetric matrix $T$ with travel time between each pair of locations, are given. Every node $\mathrm{n}_i$ has a positive score or reward, $r_i$, a visiting time window $[o_i , c_i]$ with opening time (the earliest a visit can start) and closing time (the latest time for which a visit can start), and duration of visit $d_i$. Without loss of generality, we can assume that $\mathrm{n}_1$ is the starting and ending location for every route. The objective is to find $m$ routes with the maximum possible sum of scores, without repeating visits, starting each route on or after a given time $t_{start}$ and ending before time $t_{end}$ .

Here's everything about TOPTW environment:

.. toctree::
    :maxdepth: 1
    
    agent-reward/agent-reward
    agent-selector/agent-selector
    environment/environment
    generation/generation
    observations/observations