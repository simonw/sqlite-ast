# sqlite-ast

[![PyPI](https://img.shields.io/pypi/v/sqlite-ast.svg)](https://pypi.org/project/sqlite-ast/)
[![Tests](https://github.com/simonw/sqlite-ast/actions/workflows/test.yml/badge.svg)](https://github.com/simonw/sqlite-ast/actions/workflows/test.yml)
[![Changelog](https://img.shields.io/github/v/release/simonw/sqlite-ast?include_prereleases&label=changelog)](https://github.com/simonw/sqlite-ast/releases)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/simonw/sqlite-ast/blob/main/LICENSE)

Python library for parsing SQLite SELECT queries into an AST

## Installation

Install this library using `pip`:
```bash
pip install sqlite-ast
```
## Usage

The main entry point is `parse(sql)`, which returns a nested Python dictionary:

```python
from sqlite_ast import parse

ast = parse("select a from t where a > 1 order by a desc limit 10")
```

You can pretty-print that dictionary as JSON:

```python
import json
from sqlite_ast import parse

ast = parse("select 1")
print(json.dumps(ast, indent=2))
```

If you want structured dataclass nodes instead of dictionaries, use `parse_ast(sql)`:

```python
from sqlite_ast import parse_ast

node = parse_ast("select 1")
ast = node.to_dict()
```

Parse failures raise `ParseError`:

```python
from sqlite_ast import parse, ParseError

try:
    parse("select from")
except ParseError as e:
    print(e)
```

## Development

To contribute to this library, first checkout the code. Then run the tests with [uv](https://github.com/astral-sh/uv):
```bash
cd sqlite-ast
uv run pytest
```
