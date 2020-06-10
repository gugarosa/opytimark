# Opytimark: Python Optimization Benchmarking Functions

[![Latest release](https://img.shields.io/github/release/gugarosa/opytimark.svg)](https://github.com/gugarosa/opytimark/releases)
[![Build status](https://img.shields.io/travis/com/gugarosa/opytimark/master.svg)](https://github.com/gugarosa/opytimark/releases)
[![Open issues](https://img.shields.io/github/issues/gugarosa/opytimark.svg)](https://github.com/gugarosa/opytimark/issues)
[![License](https://img.shields.io/github/license/gugarosa/opytimark.svg)](https://github.com/gugarosa/opytimark/blob/master/LICENSE)

## Welcome to Opytimark.
Did you ever need a set of pre-defined functions in order to test your optimization algorithm? Are you tired of implementing and validating by hand every function? If yes, Opytimark is the real deal! This package provides straightforward implementation of benchmarking functions for optimization tasks.

Use Opytimark if you need a library or wish to:
* Create your benchmarking function;
* Design or use pre-loaded benchmarking functions;
* Because it is overwhelming to evaluate things.

Read the docs at [opytimark.readthedocs.io](https://opytimark.readthedocs.io).

Opytimark is compatible with: **Python 3.6+**.

---

## Package guidelines

1. The very first information you need is in the very **next** section.
2. **Installing** is also easy if you wish to read the code and bump yourself into, follow along.
3. Note that there might be some **additional** steps in order to use our solutions.
4. If there is a problem, please do not **hesitate**, call us.

---

## Citation

If you use Opytimark to fulfill any of your needs, please cite us:

```
@misc{rosa2019opytimizer,
    title={Opytimizer: A Nature-Inspired Python Optimizer},
    author={Gustavo H. de Rosa and João P. Papa},
    year={2019},
    eprint={1912.13002},
    archivePrefix={arXiv},
    primaryClass={cs.NE}
}
```

---

## Getting started: 60 seconds with Opytimark

First of all. We have examples. Yes, they are commented. Just browse to `examples/`, chose your subpackage, and follow the example. We have high-level examples for most tasks we could think.

Alternatively, if you wish to learn even more, please take a minute:

Opytimark is based on the following structure, and you should pay attention to its tree:

```
- opytimark
    - core
        - benchmark
    - markers
        - boolean
        - one_dimensional
        - two_dimensional
        - many_dimensional
        - n_dimensional
    - utils
        - constants
        - decorator
        - exception
        - logging
```

### Core

Core is the core. Essentially, it is the parent of everything. You should find parent classes defining the basis of our structure. They should provide variables and methods that will help to construct other modules.

### Markers

This is why we are called Opytimark. This is the heart of the benchmarking functions, where you can find a large number of pre-defined functions. Investigate any module for more information.

### Utils

This is a utility package. Common things shared across the application should be implemented here. It is better to implement once and use as you wish than re-implementing the same thing over and over again.

---

## Installation

We believe that everything has to be easy. Not tricky or daunting, Opytimark will be the one-to-go package that you will need, from the very first installation to the daily-tasks implementing needs. If you may just run the following under your most preferred Python environment (raw, conda, virtualenv, whatever):

```Python
pip install opytimark
```

Alternatively, if you prefer to install the bleeding-edge version, please clone this repository and use:

```Python
pip install .
```

---

## Environment configuration

Note that sometimes, there is a need for additional implementation. If needed, from here, you will be the one to know all of its details.

### Ubuntu

No specific additional commands needed.

### Windows

No specific additional commands needed.

### MacOS

No specific additional commands needed.

---

## Support

We know that we do our best, but it is inevitable to acknowledge that we make mistakes. If you ever need to report a bug, report a problem, talk to us, please do so! We will be available at our bests at this repository or gustavo.rosa@unesp.br.

---