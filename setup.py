from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent
about = {}
exec((ROOT / "opytimark" / "__init__.py").read_text(encoding="utf-8"), about)

setup(
    name="opytimark",
    version=about["__version__"],
    description="Python Optimization Benchmarking Functions",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Gustavo Rosa",
    author_email="gustavo.rosa@unesp.br",
    url="https://github.com/gugarosa/opytimark",
    license="Apache 2.0",
    python_requires=">=3.6",
    install_requires=["numpy>=1.19.5"],
    extras_require={
        "tests": ["coverage", "pytest", "pytest-pep8"],
        "dev": [
            "pre-commit>=2.17.0; python_full_version >= '3.6.1'",
            "pylint>=2.7.2; python_full_version >= '3.6.2'",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=find_packages(),
    package_data={"opytimark.data": ["*.tar.gz"]},
)
