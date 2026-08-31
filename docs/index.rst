Opytimark
=========

Opytimark provides ready-to-use benchmark functions for evaluating optimization
algorithms.

Opytimark supports Python 3.11 or newer.

Install it from PyPI into a project managed by uv:

.. code-block:: console

   uv add opytimark

For a consumer installation in an existing Python environment, pip is also
supported:

.. code-block:: console

   pip install opytimark

Evaluate a benchmark:

.. code-block:: python

   import numpy as np

   from opytimark.markers.n_dimensional import Sphere

   value = Sphere()(np.array([1.0, 2.0, 3.0]))
   print(value)

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
