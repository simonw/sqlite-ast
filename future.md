# Future Steps

## Broader SQL Statement Support

The current implementation only handles SELECT statements (including compound selects, CTEs, and VALUES). Full SQLite coverage would require:

### INSERT
- `INSERT INTO table VALUES (...)` / `INSERT INTO table SELECT ...`
- `INSERT OR REPLACE`, `INSERT OR IGNORE`, etc.
- `RETURNING` clause (SQLite 3.35+)
- Upsert: `ON CONFLICT DO NOTHING / DO UPDATE SET ...`

### UPDATE
- `UPDATE table SET col = expr WHERE ...`
- `UPDATE FROM` (SQLite 3.33+)
- `UPDATE OR REPLACE / IGNORE / etc.`
- `RETURNING` clause
- `LIMIT` and `ORDER BY` on UPDATE

### DELETE
- `DELETE FROM table WHERE ...`
- `RETURNING` clause
- `LIMIT` and `ORDER BY` on DELETE

### CREATE TABLE
- Column definitions with types, constraints, defaults
- `PRIMARY KEY`, `UNIQUE`, `NOT NULL`, `CHECK`, `FOREIGN KEY`
- `WITHOUT ROWID`, `STRICT` tables
- `CREATE TABLE AS SELECT`
- `IF NOT EXISTS`

### CREATE INDEX
- `CREATE [UNIQUE] INDEX [IF NOT EXISTS] ...`
- Expression indexes
- Partial indexes (`WHERE` clause)

### CREATE VIEW
- `CREATE [TEMP] VIEW [IF NOT EXISTS] name AS SELECT ...`

### CREATE TRIGGER
- `BEFORE / AFTER / INSTEAD OF` triggers
- `INSERT / UPDATE / DELETE` events
- `FOR EACH ROW`
- Trigger body (multiple statements)

### DROP statements
- `DROP TABLE / INDEX / VIEW / TRIGGER [IF EXISTS]`

### ALTER TABLE
- `RENAME TO`, `RENAME COLUMN`, `ADD COLUMN`, `DROP COLUMN`

### Other statements
- `EXPLAIN [QUERY PLAN]`
- `ATTACH / DETACH DATABASE`
- `BEGIN / COMMIT / ROLLBACK / SAVEPOINT / RELEASE`
- `PRAGMA`
- `VACUUM`
- `REINDEX`
- `ANALYZE`

## Expression Coverage Gaps

The current expression parser covers the conformance test suite but may not handle all possible expressions:

### Not yet tested
- `REGEXP` operator (requires user-defined function, but syntactically valid)
- `IS DISTINCT FROM` / `IS NOT DISTINCT FROM` (parsed but not tested)
- `NOT EXISTS` (parsed but not in conformance tests)
- `IN ()` — empty IN list (SQLite folds to constant 0/1)
- `LIKE` with `ESCAPE` clause — parsed, only tested in kitchen_sink
- `RAISE(IGNORE)`, `RAISE(ROLLBACK, msg)`, etc. (trigger expressions)
- `IIF(cond, true_val, false_val)` — just a regular function call, should work

### Multi-word type names in CAST
- `CAST(x AS VARCHAR(255))` — type names with parenthesized parameters
- `CAST(x AS UNSIGNED BIG INT)` — multi-word type names
- Currently only single-word type names are parsed

### Expression edge cases
- Double-quoted string literals (SQLite treats them as identifiers first, strings as fallback)
- Unary plus collapse: `+(+x)` folds to `+(x)` in SQLite (not tested)
- Very large integer literals that don't fit in int32 (stored as text, not EP_IntValue)
- Nested parenthesized expressions with trailing operators

## Conformance Test Expansion

The current 90 tests cover the major features but there are areas worth expanding:

### Additional test categories to propose for sqlite-ast-conformance
- Schema-qualified table names: `SELECT * FROM main.foo`
- Table-valued functions: `SELECT * FROM json_each('[1,2,3]')`
- Indexed BY / NOT INDEXED hints
- Multiple column aliases without AS keyword
- Complex nested CTEs (CTE referencing another CTE)
- Window functions with EXCLUDE clause
- Window functions with GROUPS frame type
- LIKE with ESCAPE in WHERE clause
- IN with empty list
- IS DISTINCT FROM / IS NOT DISTINCT FROM
- Multiple USING columns: `JOIN ... USING (a, b)`
- Implicit aliases (without AS keyword) in various contexts

## API Improvements

### Visitor pattern
Add a visitor/transformer API for walking and modifying AST nodes:
```python
from python_sqlite_ast import parse_ast, Visitor

class ColumnCollector(Visitor):
    def __init__(self):
        self.columns = []

    def visit_Name(self, node):
        self.columns.append(node.name)

ast = parse_ast("SELECT a, b FROM foo WHERE c > 1")
collector = ColumnCollector()
collector.visit(ast)
# collector.columns == ["a", "b", "c"]
```

### AST-to-SQL serializer
Round-trip capability: convert AST back to SQL string.
```python
from python_sqlite_ast import parse_ast, to_sql

ast = parse_ast("SELECT * FROM foo WHERE x > 5")
sql = to_sql(ast)
# "SELECT * FROM foo WHERE x > 5"
```

### Query rewriting
Higher-level utilities for common query transformations:
- Add/remove WHERE conditions
- Add/remove columns
- Add/remove JOINs
- Convert subqueries to CTEs

### Type stubs / better typing
- Add `py.typed` marker
- Improve type annotations on AST node fields (currently many are `Any`)
- Consider using a base class or Protocol for expression nodes

## Performance Optimization

The current implementation is straightforward but could be optimized:

- **Lazy tokenization**: Tokenize on-demand instead of all-at-once
- **String interning**: Intern commonly-used keyword strings
- **Slot-based dataclasses**: Already using `@dataclass(slots=True)` for Token; could extend to AST nodes
- **Compiled regex for tokenizer**: The character-by-character tokenizer could use regex for number/identifier patterns

## Testing Improvements

- **Fuzz testing**: Use Hypothesis or a grammar-based fuzzer to generate random valid SELECT statements and verify parsing doesn't crash
- **Round-trip testing**: Parse → serialize → parse → compare ASTs
- **Error message quality**: Test that parse errors have useful position and context information
- **Property-based tests**: Verify structural invariants (e.g., every InList has either values or select, never both)

## Packaging and Distribution

- Add `py.typed` marker for PEP 561
- Publish to PyPI
- Set up CI (GitHub Actions) running the conformance tests
- Add benchmarks comparing parse speed to other Python SQL parsers
