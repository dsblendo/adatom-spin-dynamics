# adatom-spin-dynamics

Repo to model the spin dynamics of a single adatom.

# Development Setup

## How to begin utilizing this repo

1. Clone the repository
2. Ensure pyenv and poetry are installed on your box
3. If Python 3.8.16 is not already installed via pyenv, run `pyenv install 3.8.16` (for latest version see .python_version file)
4. `cd adatom-spin-dynamics`
5. `pyenv local 3.8.16`
6. `python3 -m venv .venv` followed by `source .venv/bin/activate`
7. Install required packages (this will install pre-commit required in the next step): `poetry install`
8. `pre-commit install` . This will ensure pre-commit hook checks are run before committing.
