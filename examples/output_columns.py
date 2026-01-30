from sqlite_ast import parse_ast

node = parse_ast("SELECT u.name, o.total FROM users u JOIN orders o ON 1=1")
for col in node.output_columns():
    print(col, col.table, col.column)
