# scvi-criticism

[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/YosefLab/scvi-criticism/test.yaml?branch=main
[link-tests]: https://github.com/YosefLab/scvi-criticism/actions/workflows/test.yml
[badge-docs]: https://img.shields.io/readthedocs/scvi-criticism

Evaluation metrics for scvi-tools models


# ⚠️ !! WE'VE MOVED !! ⚠️   
This package is now merged into the scvi-tools package and is accessible via the `scvi.criticism` submodule.  
As part of the transition, we've also removed the generic plotting utilities. We'll give some plotting examples in upcoming scvi tutorials but otherwise will leave the freedom to end-users to use their custom plotting for their specific use case(s).  
Thanks for using this tool and please leave any future feedback under scvi-tools [issues](https://github.com/scverse/scvi-tools/issues) as this repo will be archived.


**OLD INSTRUCTIONS** 👇

## Getting started

Please refer to the [documentation][link-docs].

## Installation

You need to have Python 3.8 or newer installed on your system. If you don't have
Python installed, we recommend installing [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

1. Install the latest release of `scvi-criticism` from `PyPI <https://pypi.org/project/scvi-criticism/>`\_:

```bash
pip install scvi-criticism
```

2. Install the latest development version:

```bash
pip install git+https://github.com/YosefLab/scvi-criticism.git@main
```

## Release notes

See the [changelog][changelog].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][scverse-discourse].
If you found a bug, please use the [issue tracker][issue-tracker].

## Citations

N/A

## Acknowledgements

Some of the code in this package was adapted from draft code authored by [@adamgayoso](https://github.com/adamgayoso).

[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/YosefLab/scvi-criticism/issues
[changelog]: https://scvi-criticism.readthedocs.io/en/latest/changelog.html
[link-docs]: https://scvi-criticism.readthedocs.io~
