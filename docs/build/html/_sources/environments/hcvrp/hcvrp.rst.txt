:hide-toc:

==========
HCVRP
==========

Heterogeneous Capacitated Vehicle Routing Problem (HCVRP)

In the HCVRP, a fleet of heterogeneous vehicles is dispatched from a depot to serve a set of customers with known demands. Unlike the standard CVRP, each vehicle has its own individual capacity and travel speed. Each customer can only be served once by a single vehicle, and each vehicle cannot exceed its individual capacity. Vehicles must start and end their routes at the depot.

The objective is to minimize the total combined traveled time of all vehicle routes.

Here's everything about HCVRP environment:

.. toctree::
    :maxdepth: 1
    
    agent-reward/agent-reward
    agent-selector/agent-selector
    environment/environment
    generation/generation
    observations/observations
