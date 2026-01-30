from sqlite_ast import parse_ast

node = parse_ast("""
    SELECT UPPER(name), COUNT(*), ROUND(AVG(total), 2)
    FROM users u
    JOIN orders o ON u.id = o.user_id
    GROUP BY name
""")
print(node.functions_used())
