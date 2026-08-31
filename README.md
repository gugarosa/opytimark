# Opytimark: Python Optimization Benchmarking Functions

[![Latest release](https://img.shields.io/github/release/gugarosa/opytimark.svg)](https://github.com/gugarosa/opytimark/releases)
[![CI](https://github.com/gugarosa/opytimark/actions/workflows/ci.yml/badge.svg)](https://github.com/gugarosa/opytimark/actions/workflows/ci.yml)
[![Documentation](https://readthedocs.org/projects/opytimark/badge/?version=latest)](https://opytimark.readthedocs.io)
[![Open issues](https://img.shields.io/github/issues/gugarosa/opytimark.svg)](https://github.com/gugarosa/opytimark/issues)
[![License](https://img.shields.io/github/license/gugarosa/opytimark.svg)](https://github.com/gugarosa/opytimark/blob/main/LICENSE)

Opytimark provides ready-to-use benchmark functions for evaluating optimization
algorithms.

Opytimark supports Python 3.11 or newer. Read the full API reference at
[opytimark.readthedocs.io](https://opytimark.readthedocs.io).

## Installation

Opytimark is published on PyPI. Add it to a project managed by uv with:

```bash
uv add opytimark
```

For a consumer installation in an existing Python environment, pip is also supported:

```bash
pip install opytimark
```

## Usage

```python
import numpy as np

from opytimark.markers.n_dimensional import Sphere

value = Sphere()(np.array([1.0, 2.0, 3.0]))
print(value)
```

More examples are available in [`examples/`](examples).

## Development

Install [uv](https://docs.astral.sh/uv/), clone the repository, then run:

```bash
uv sync --locked
uv run pytest
uv run pre-commit run --all-files
uv run --locked --group docs sphinx-build -W --keep-going -b html docs docs/_build/html
uv build
```

## Citation

If you use Opytimark, please cite:

```bibtex
@misc{rosa2019opytimizer,
    title={Opytimizer: A Nature-Inspired Python Optimizer},
    author={Gustavo H. de Rosa and João P. Papa},
    year={2019},
    eprint={1912.13002},
    archivePrefix={arXiv},
    primaryClass={cs.NE}
}
```
