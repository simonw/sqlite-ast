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
