from sqlite_ast import parse_ast

SCHEMA = {
    "users": ["id", "name", "email"],
    "orders": ["id", "user_id", "total"],
}

node = parse_ast("SELECT * FROM users")
print(node.output_columns(lambda table: SCHEMA.get(table, [])))
