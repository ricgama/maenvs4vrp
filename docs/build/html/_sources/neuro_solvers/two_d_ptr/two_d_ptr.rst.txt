:hide-toc:

========================
2D-Ptr
========================

Implementation of the `2D Pointer Network <https://arxiv.org/abs/2401.00185>`_ for vehicle routing problems. This model jointly selects the next node and the next agent at each step, enabling full multi-agent coordination via a two-dimensional pointer mechanism.

**References**

* Fang, K., Ge, D., & Li, Y. (2024). *2D-Ptr: 2D Pointer Network for Multi-Agent VRP.* `arXiv:2401.00185 <https://arxiv.org/abs/2401.00185>`_
* Kool, W., van Hoof, H., & Welling, M. (2019). *Attention, Learn to Solve Routing Problems!* `arXiv:1803.08475 <https://arxiv.org/abs/1803.08475>`_

Policy Network
------------------

Defined in ``td_pointer_model.py``.

.. autoclass:: maenvs4vrp.neuro_solvers.two_d_ptr.td_pointer_model.AttentionScore
    :members:
    :special-members: __init__

.. autoclass:: maenvs4vrp.neuro_solvers.two_d_ptr.td_pointer_model.MultiHeadAttention
    :members:
    :special-members: __init__

.. autoclass:: maenvs4vrp.neuro_solvers.two_d_ptr.td_pointer_model.MultiHeadAttentionProj
    :members:
    :special-members: __init__

.. autoclass:: maenvs4vrp.neuro_solvers.two_d_ptr.td_pointer_model.SkipConnection
    :members:
    :special-members: __init__

.. autoclass:: maenvs4vrp.neuro_solvers.two_d_ptr.td_pointer_model.Normalization
    :members:
    :special-members: __init__

.. autoclass:: maenvs4vrp.neuro_solvers.two_d_ptr.td_pointer_model.MultiHeadAttentionLayer
    :members:
    :special-members: __init__

.. autoclass:: maenvs4vrp.neuro_solvers.two_d_ptr.td_pointer_model.GraphAttentionEncoder
    :members:
    :special-members: __init__

.. autoclass:: maenvs4vrp.neuro_solvers.two_d_ptr.td_pointer_model.DynamicEmbedding
    :members:
    :special-members: __init__

.. autoclass:: maenvs4vrp.neuro_solvers.two_d_ptr.td_pointer_model.PolicyNet
    :members:
    :special-members: __init__

Training Scripts
------------------

REINFORCE + POMO Shared Baseline (``train_reinforce_shared_baseline.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Trains the 2D-Ptr policy using the REINFORCE algorithm with a POMO-style shared baseline derived from instance augmentation.
