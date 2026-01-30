# python-sqlite-ast

[![PyPI](https://img.shields.io/pypi/v/python-sqlite-ast.svg)](https://pypi.org/project/python-sqlite-ast/)
[![Tests](https://github.com/simonw/python-sqlite-ast/actions/workflows/test.yml/badge.svg)](https://github.com/simonw/python-sqlite-ast/actions/workflows/test.yml)
[![Changelog](https://img.shields.io/github/v/release/simonw/python-sqlite-ast?include_prereleases&label=changelog)](https://github.com/simonw/python-sqlite-ast/releases)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/simonw/python-sqlite-ast/blob/main/LICENSE)

Python library for parsing SQLite SELECT queries into an AST

## Installation

Install this library using `pip`:
```bash
pip install python-sqlite-ast
```
## Usage

Usage instructions go here.

## Development

To contribute to this library, first checkout the code. Then create a new virtual environment:
```bash
cd python-sqlite-ast
python -m venv venv
source venv/bin/activate
```
Now install the dependencies and test dependencies:
```bash
python -m pip install -e '.[test]'
```
To run the tests:
```bash
python -m pytest
```
