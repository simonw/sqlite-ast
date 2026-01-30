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
## Demo

You can try this library out in your browser (via Pyodide) at [tools.simonwillison.net/sqlite-ast](https://tools.simonwillison.net/sqlite-ast).

## Usage

The main entry point is `parse(sql)`, which returns a nested Python dictionary:

<!-- [[[cog
import io
from pathlib import Path
from contextlib import redirect_stdout

def render_example(path_str: str, output_lang: str = "text") -> None:
    path = Path(path_str)
    code = path.read_text().rstrip()

    cog.outl("```python")
    cog.outl(code)
    cog.outl("```")

    buf = io.StringIO()
    with redirect_stdout(buf):
        exec(compile(code, str(path), "exec"), {})

    cog.outl(f"```{output_lang}")
    cog.outl(buf.getvalue().rstrip())
    cog.outl("```")

render_example("examples/parse_basic.py", "text")
]]] -->
```python
from sqlite_ast import parse

ast = parse("select 1")
print(ast)
```
```text
{'type': 'select', 'distinct': False, 'all': False, 'columns': [{'expr': {'type': 'integer', 'value': 1}, 'alias': None}], 'from': None, 'where': None, 'group_by': None, 'having': None, 'order_by': None, 'limit': None}
```
<!-- [[[end]]] -->

You can pretty-print that dictionary as JSON:

<!-- [[[cog
render_example("examples/parse_json.py", "json")
]]] -->
```python
import json
from sqlite_ast import parse

ast = parse("select 1")
print(json.dumps(ast, indent=2))
```
```json
{
  "type": "select",
  "distinct": false,
  "all": false,
  "columns": [
    {
      "expr": {
        "type": "integer",
        "value": 1
      },
      "alias": null
    }
  ],
  "from": null,
  "where": null,
  "group_by": null,
  "having": null,
  "order_by": null,
  "limit": null
}
```
<!-- [[[end]]] -->

If you want structured dataclass nodes instead of dictionaries, use `parse_ast(sql)`:

<!-- [[[cog
render_example("examples/parse_ast.py", "text")
]]] -->
```python
from pprint import pprint
from sqlite_ast import parse_ast

node = parse_ast("select 1")
pprint(node)
```
```text
Select(distinct=False,
       all=False,
       with_ctes=None,
       columns=[ResultColumn(expr=IntegerLiteral(value=1), alias=None)],
       from_clause=None,
       where=None,
       group_by=None,
       having=None,
       window_definitions=None,
       order_by=None,
       limit=None,
       offset=None,
       _has_limit=False,
       _compound_member=False)
```
<!-- [[[end]]] -->

Parse failures raise `ParseError`:

<!-- [[[cog
render_example("examples/parse_error.py", "text")
]]] -->
```python
from pprint import pprint
from sqlite_ast import parse, ParseError

try:
    parse("select 1 union select")
except ParseError as e:
    print(e)
    print("\nPartial AST:")
    pprint(e.partial_ast)
```
```text
Parse error at position 21: Unexpected token in expression: EOF ('')

Partial AST:
Select(distinct=False,
       all=False,
       with_ctes=None,
       columns=[ResultColumn(expr=IntegerLiteral(value=1), alias=None)],
       from_clause=None,
       where=None,
       group_by=None,
       having=None,
       window_definitions=None,
       order_by=None,
       limit=None,
       offset=None,
       _has_limit=False,
       _compound_member=True)
```
<!-- [[[end]]] -->

## Analysis methods

AST nodes returned by `parse_ast()` provide methods for extracting high-level metadata from queries.

### `tables_referenced()`

Returns all table names referenced anywhere in the query (FROM, JOIN, subqueries, CTEs):

<!-- [[[cog
render_example("examples/tables_referenced.py", "text")
]]] -->
```python
from sqlite_ast import parse_ast

node = parse_ast("""
    SELECT u.name, o.total
    FROM users u
    JOIN orders o ON u.id = o.user_id
    WHERE o.total > (SELECT AVG(total) FROM orders)
""")
print(node.tables_referenced())
```
```text
['users', 'orders']
```
<!-- [[[end]]] -->

### `functions_used()`

Returns all SQL function names used in the query:

<!-- [[[cog
render_example("examples/functions_used.py", "text")
]]] -->
```python
from sqlite_ast import parse_ast

node = parse_ast("""
    SELECT UPPER(name), COUNT(*), ROUND(AVG(total), 2)
    FROM users u
    JOIN orders o ON u.id = o.user_id
    GROUP BY name
""")
print(node.functions_used())
```
```text
['UPPER', 'COUNT', 'ROUND', 'AVG']
```
<!-- [[[end]]] -->

### `output_columns()`

Returns the columns produced by a SELECT as a list of `OutputColumn` objects, each with `.table` and `.column` attributes:

<!-- [[[cog
render_example("examples/output_columns.py", "text")
]]] -->
```python
from sqlite_ast import parse_ast

node = parse_ast("SELECT u.name, o.total FROM users u JOIN orders o ON 1=1")
for col in node.output_columns():
    print(col, col.table, col.column)
```
```text
users.name users name
orders.total orders total
```
<!-- [[[end]]] -->

Pass a `columns_for_table` callback to resolve `SELECT *`, `SELECT t.*`, and bare column names to their owning table:

<!-- [[[cog
render_example("examples/output_columns_bare.py", "text")
]]] -->
```python
from sqlite_ast import parse_ast

SCHEMA = {
    "users": ["id", "name", "email"],
    "orders": ["id", "user_id", "total"],
}

node = parse_ast("SELECT id, name, email FROM users")
for col in node.output_columns(lambda table: SCHEMA.get(table, [])):
    print(col)
```
```text
users.id
users.name
users.email
```
<!-- [[[end]]] -->

The same callback resolves `SELECT *` expansion:

<!-- [[[cog
render_example("examples/output_columns_star.py", "text")
]]] -->
```python
from sqlite_ast import parse_ast

SCHEMA = {
    "users": ["id", "name", "email"],
    "orders": ["id", "user_id", "total"],
}

node = parse_ast("SELECT * FROM users")
print(node.output_columns(lambda table: SCHEMA.get(table, [])))
```
```text
[OutputColumn(table='users', column='id'), OutputColumn(table='users', column='name'), OutputColumn(table='users', column='email')]
```
<!-- [[[end]]] -->

CTE and subquery-in-FROM columns are inferred automatically:

<!-- [[[cog
render_example("examples/output_columns_cte.py", "text")
]]] -->
```python
from sqlite_ast import parse_ast

SCHEMA = {
    "orders": ["id", "user_id", "total"],
}

node = parse_ast("""
    WITH totals AS (
        SELECT user_id, SUM(total) AS total_spent
        FROM orders GROUP BY user_id
    )
    SELECT * FROM totals
""")
print(node.output_columns(lambda table: SCHEMA.get(table, [])))
```
```text
[OutputColumn(table='totals', column='user_id'), OutputColumn(table='totals', column='total_spent')]
```
<!-- [[[end]]] -->

## Development

To contribute to this library, first checkout the code. Then run the tests with [uv](https://github.com/astral-sh/uv):
```bash
cd sqlite-ast
uv run pytest
```
