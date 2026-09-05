Numerical behavior
==================

Reproducibility
---------------

Opytimark does not seed NumPy when imported. Noisy functions and randomized
grouped CEC functions continue to use NumPy's global random state. Seed it
explicitly before an experiment when reproducibility is required:

.. code-block:: python

   import numpy as np

   from opytimark.markers.n_dimensional import Quartic

   np.random.seed(0)
   value = Quartic()(np.zeros(10))

Some grouped CEC functions draw a new permutation on every evaluation, not just
at construction. Repeating a point does not therefore imply repeating its
fitness. This existing permutation policy is unchanged.

Version 3.0.1 corrections
-------------------------

The following corrections intentionally change affected fitness values relative
to version 3.0.0. Compare optimization runs using the same Opytimark revision.
Constructor arguments, benchmark names, metadata, and supported interpreters
remain unchanged.

* Elliptic coefficients range from 1 to :math:`10^6`, not :math:`10^7`.
  The standalone benchmark and CEC 2005 F3, 2010 F1, and 2013 F1 share this
  calculation. For one variable, the coefficient is one.
* CEC 2010 F4--F8 use the documented :math:`10^6` group multiplier. Rotated
  groups select both axes of the supplied matrix when ``group_size`` is smaller
  than the matrix dimension.
* CEC 2013 diagonal conditioning uses
  :math:`\alpha^{i/(2(D-1))}` for zero-based :math:`i`. Its first coefficient
  is one, so the first coordinate is no longer discarded. The one-dimensional
  transform is the identity. ``T_diagonal`` still returns a matrix; benchmark
  implementations apply its diagonal directly without allocating a dense matrix.
* CEC 2005 composition weights are normalized after attenuating non-maximal
  weights. Relative exponentials avoid all weights underflowing to zero.
  F17's noise, F20's boundary shift, and F23's discontinuity are retained.
* The asymmetry transform evaluates square roots only for positive coordinates;
  it neither suppresses floating-point errors nor mutates the input.

The conditioning definitions are described in the
`CEC 2013 large-scale technical report
<https://www.al-roomi.org/multimedia/CEC_Database/CEC2013/LargeScaleGlobalOptimization/CEC2013_LargeScaleGO_TechnicalReport.pdf>`_;
hybrid composition is described in the
`CEC 2005 technical report
<https://www.al-roomi.org/multimedia/CEC_Database/CEC2005/CEC2005_TechnicalReport.pdf>`_.
These focused corrections are not a certification of full reference-suite
conformance for every benchmark.

Data and failures
-----------------

Bundled CEC data remains available without a network connection. Local overrides
still take precedence. New downloads and extractions become visible at their
final paths only after completion, so an interrupted attempt can be retried.
Existing files and extracted folders are not refreshed or overwritten by a
normal cache hit.

``untar_file`` is for trusted archives; its existing path check is not a general
archive-sandboxing guarantee. Python's native extraction filter is not enabled
unconditionally because the project also supports Python 3.11.0--3.11.3, which
predate that API.
