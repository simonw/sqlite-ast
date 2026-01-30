from sqlite_ast import parse_ast

node = parse_ast("""
    SELECT u.name, o.total
    FROM users u
    JOIN orders o ON u.id = o.user_id
    WHERE o.total > (SELECT AVG(total) FROM orders)
""")
print(node.tables_referenced())
