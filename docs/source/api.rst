.. This file provides the instructions for how to display the API documentation generated using sphinx autodoc
   extension. Use it to declare Python documentation sub-directories via appropriate modules (autodoc, etc.).

Reference Command Line Interfaces
=================================

.. automodule:: sl_experiment.cli
   :members:
   :undoc-members:
   :show-inheritance:

.. click:: sl_experiment.cli:calculate_crc
   :prog: sl-crc
   :nested: full

.. click:: sl_experiment.cli:list_devices
   :prog: sl-devices
   :nested: full

.. click:: sl_experiment.cli:maintain_vr
   :prog: sl-maintain-vr
   :nested: full

.. click:: sl_experiment.cli:lick_training
   :prog: sl-lick-train
   :nested: full

.. click:: sl_experiment.cli:run_training
   :prog: sl-run-train
   :nested: full

.. click:: sl_experiment.cli:run_experiment
   :prog: sl-experiment
   :nested: full

.. click:: sl_experiment.cli:preprocess_session
   :prog: sl-experiment
   :nested: full

.. click:: sl_experiment.cli:purge_data
   :prog: sl-experiment
   :nested: full

.. click:: sl_experiment.cli:replace_local_root_directory
   :prog: sl-experiment
   :nested: full

Experiment Interfaces
=====================
.. automodule:: sl_experiment.experiment
   :members:
   :undoc-members:
   :show-inheritance:

Ataraxis Binding Classes
========================
.. automodule:: sl_experiment.binding_classes
   :members:
   :undoc-members:
   :show-inheritance:

Runtime Data Visualizers
========================
.. automodule:: sl_experiment.visualizers
   :members:
   :undoc-members:
   :show-inheritance:

Zaber Interfaces
================
.. automodule:: sl_experiment.zaber_bindings
   :members:
   :undoc-members:
   :show-inheritance:

AXMC Module Interfaces
======================
.. automodule:: sl_experiment.module_interfaces
   :members:
   :undoc-members:
   :show-inheritance:

Google Sheet Tools
==================
.. automodule:: sl_experiment.google_sheet_tools
   :members:
   :undoc-members:
   :show-inheritance:

Packaging Tools
===============
.. automodule:: sl_experiment.packaging_tools
   :members:
   :undoc-members:
   :show-inheritance:

Transfer Tools
==============
.. automodule:: sl_experiment.transfer_tools
   :members:
   :undoc-members:
   :show-inheritance:

Data Preprocessing Tools
========================
.. automodule:: sl_experiment.data_preprocessing
   :members:
   :undoc-members:
   :show-inheritance:

Configuration and Data Storage Classes
======================================
.. automodule:: sl_experiment.data_classes
   :members:
   :undoc-members:
   :show-inheritance: