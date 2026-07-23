:hide-toc:

==========
PCVRP
==========

Prize-Collecting Capacitated Vehicle Routing Problem (PCVRP)

In the PCVRP, a fleet of vehicles is dispatched from a depot to collect profits from a set of customers. Not all customers need to be visited; the objective is to maximize the total collected profit while respecting vehicle capacity constraints. Each customer can be visited at most once by a single vehicle, and each vehicle must return to the depot when its capacity is exhausted.

The objective is to maximize the total profit collected across all vehicle routes.

Here's everything about PCVRP environment:

.. toctree::
    :maxdepth: 1
    
    agent-reward/agent-reward
    agent-selector/agent-selector
    environment/environment
    generation/generation
    observations/observations
