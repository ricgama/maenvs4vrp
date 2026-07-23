:hide-toc:

========================
Attention Model (AM)
========================

Implementation of the `Attention Model <https://arxiv.org/abs/1803.08475>`_ for vehicle routing problems, using the POMO shared-baseline augmentation for training.

**References**

* Kool, W., van Hoof, H., & Welling, M. (2019). *Attention, Learn to Solve Routing Problems!* ICLR 2019. `arXiv:1803.08475 <https://arxiv.org/abs/1803.08475>`_
* Kwon, Y. D., Choo, J., Kim, B., Yoon, I., Gwon, Y., & Min, S. (2020). *POMO: Policy Optimization with Multiple Optima for Reinforcement Learning.* NeurIPS 2020. `arXiv:2010.16011 <https://arxiv.org/abs/2010.16011>`_

Policy Network
------------------

Defined in ``policy_net_am.py``.

.. autoclass:: maenvs4vrp.neuro_solvers.attention_model.policy_net_am.AttentionScore
    :members:
    :special-members: __init__

.. autoclass:: maenvs4vrp.neuro_solvers.attention_model.policy_net_am.MultiHeadAttention
    :members:
    :special-members: __init__

.. autoclass:: maenvs4vrp.neuro_solvers.attention_model.policy_net_am.MultiHeadAttentionProj
    :members:
    :special-members: __init__

.. autoclass:: maenvs4vrp.neuro_solvers.attention_model.policy_net_am.SkipConnection
    :members:
    :special-members: __init__

.. autoclass:: maenvs4vrp.neuro_solvers.attention_model.policy_net_am.Normalization
    :members:
    :special-members: __init__

.. autoclass:: maenvs4vrp.neuro_solvers.attention_model.policy_net_am.MultiHeadAttentionLayer
    :members:
    :special-members: __init__

.. autoclass:: maenvs4vrp.neuro_solvers.attention_model.policy_net_am.GraphAttentionEncoder
    :members:
    :special-members: __init__

.. autoclass:: maenvs4vrp.neuro_solvers.attention_model.policy_net_am.DynamicEmbedding
    :members:
    :special-members: __init__

.. autoclass:: maenvs4vrp.neuro_solvers.attention_model.policy_net_am.PolicyNet
    :members:
    :special-members: __init__

.. autoclass:: maenvs4vrp.neuro_solvers.attention_model.policy_net_am.ActionCriticNet
    :members:
    :special-members: __init__

Training Scripts
------------------

Three training algorithms are provided.

REINFORCE + POMO Shared Baseline (``train_reinforce_shared_baseline.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Trains the AM policy using the REINFORCE algorithm with a POMO-style shared baseline derived from instance augmentation.

PPO (``train_ppo.py``)
~~~~~~~~~~~~~~~~~~~~~~~

Trains an actor-critic AM network using Proximal Policy Optimization (PPO). Adapted from `CleanRL <https://github.com/vwxyzjn/cleanrl>`_.

GRPO (``train_grpo.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~

Trains the AM policy using Group Relative Policy Optimization (GRPO), a clipped PPO-style objective where advantages are computed relative to a group of augmented rollouts of the same instance. No separate critic is required.
