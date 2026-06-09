.. _user_examples:

Examples
========

This section provides practical, hands-on examples to help you get the most out of SAFIRE.
Whether you want to see how a complete workflow unfolds or just need a quick reminder of how 
to perform a single task, you'll find both here. 

.. _walkthrough_examples:

Walkthroughs
============

Step-by-step walkthroughs that carry you from start to finish through a complete process. 
These show how multiple features fit together in real-world scenarios, 
making them useful for understanding context and best practices.
The walkthroughs are organized based on their application domain.

Quantum Chemistry / Molecules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. include:: examples/molecules/index.rst


Ab initio solids
~~~~~~~~~~~~~~~~

We use Quantum Espresso to perform
DFT calculations, and Coquí to generate a Hamiltonian and write trial wavefunctions.

1. `Charge density in Si - CoQuí interface <https://colab.research.google.com/drive/1V2qDFA5PJkM2xeliDLsKuTrWR1FpqTal>`_
2. `Carbon diamond PySCF interface <https://colab.research.google.com/drive/172tZtwLFCHI-bW6eIVXuLRgnJqPun8z6>`_
3. `Solid Na momentum distribution <https://colab.research.google.com/drive/1qNBPcToh58UJPeg9Qx6YxcIS6RPL5a2o>`_
4. `Bulk modulous of MnO <https://colab.research.google.com/drive/1oJRRBTptrW_66NRWs7CdQix68Qgz9WuX>`_
5. `Band gap of LiH <https://colab.research.google.com/drive/17x0SnUEhi-yEuyk_gMFbIGNr_SWGUi9g>`_
6. `Magnetization of ? <https://colab.research.google.com/drive/1In-gtDJ6Ud3MwHJEJdQS-c66Y6F7wpud>`_


Lattice models
~~~~~~~~~~~~~~

We use autoHF to perform
HF calculations, and afqmctools to generate Hamiltonians and write trial wavefunctions.

1. `4x4 Hubbard Model with open shell <https://colab.research.google.com/drive/1s9bdH7XgEj4qP982ZZTOGSsQkeMgAJdX>`_
2. `Stripe Ordering <https://colab.research.google.com/drive/1HrfXBp0SkiGzYWvKKJvKBQwdL-psWV-n>`_
3. `Hubbard at half-filling <https://colab.research.google.com/drive/1MjA5PnZC5V1qRjINxXdqbs1majV_FTWf>`_
4. `pair correlation functions <https://colab.research.google.com/drive/1rI5zbAB6tMAFJWRGOp1zxIpDAZIh9LTW>`_
5. `Multi-Slater determinant trial wavefunction <https://colab.research.google.com/drive/1yTDk1u8Ww-c1t6kUKhBkEAKUIsnmGm3e>`_
6. `Hubbard on a Honeycomb lattice <https://colab.research.google.com/drive/1g9_84kc92vAyDfKHTjDiuWHXeJsp12vZ>`_
7. `Hubbard with t-prime <https://colab.research.google.com/drive/1eR1RNS-BiMToOswQ4OyJWeijcpW21a4J>`_
8. `Emery Model <https://colab.research.google.com/drive/1-zKd4kHd6Pu8WrDFbQzv3MqeyC4tiiQz>`_


.. _snippet_examples:

Snippets
========

Concise, targeted examples that focus on one specific action, option, or setting.
Snippets are designed as quick references — you’ll find exactly how to do it without extra explanation.

.. toctree::
  :maxdepth: 1
  :caption: Examples:

  snippets/01_setting_up/setup_examples
  snippets/02_running_afqmc/afqmc_examples
  snippets/03_analyzing_results/analysis_examples


