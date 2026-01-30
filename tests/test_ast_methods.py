"""
Tests for analysis methods on AST nodes:
  - tables_referenced()
  - functions_used()
  - output_columns(columns_for_table)
"""

from sqlite_ast import parse_ast
from sqlite_ast.ast_nodes import (
    Select, Compound, OutputColumn,
    IntegerLiteral, Name, ExprBase,
)


SCHEMA = {
    "users": ["id", "name", "email", "age", "created_at"],
    "orders": ["id", "user_id", "product_id", "quantity", "total", "created_at"],
    "products": ["id", "name", "price", "category", "description"],
    "categories": ["id", "name", "parent_id"],
    "reviews": ["id", "user_id", "product_id", "rating", "comment", "created_at"],
}


def columns_for_table(table: str) -> list[str]:
    return SCHEMA.get(table, [])


# ---------------------------------------------------------------------------
# ExprBase
# ---------------------------------------------------------------------------


class TestExprBase:
    def test_leaf_nodes_inherit_from_expr_base(self):
        assert issubclass(IntegerLiteral, ExprBase)
        assert issubclass(Name, ExprBase)

    def test_default_tables_referenced(self):
        node = IntegerLiteral(42)
        assert node.tables_referenced() == []

    def test_default_functions_used(self):
        node = Name("foo")
        assert node.functions_used() == []


# ---------------------------------------------------------------------------
# tables_referenced()
# ---------------------------------------------------------------------------


class TestTablesReferenced:
    def test_simple_select(self):
        node = parse_ast("SELECT name FROM users")
        assert node.tables_referenced() == ["users"]

    def test_join(self):
        node = parse_ast("""
            SELECT u.name, o.total
            FROM users u
            JOIN orders o ON u.id = o.user_id
        """)
        assert node.tables_referenced() == ["users", "orders"]

    def test_subquery_in_where(self):
        node = parse_ast("""
            SELECT name FROM users
            WHERE id IN (SELECT user_id FROM orders WHERE total > 500)
        """)
        tables = node.tables_referenced()
        assert "users" in tables
        assert "orders" in tables

    def test_exists_subquery(self):
        node = parse_ast("""
            SELECT u.name FROM users u
            WHERE EXISTS (SELECT 1 FROM reviews r WHERE r.user_id = u.id)
        """)
        tables = node.tables_referenced()
        assert "users" in tables
        assert "reviews" in tables

    def test_cte(self):
        node = parse_ast("""
            WITH big_spenders AS (
                SELECT user_id FROM orders WHERE total > 1000
            )
            SELECT u.name FROM users u
            JOIN big_spenders bs ON u.id = bs.user_id
        """)
        tables = node.tables_referenced()
        assert "orders" in tables
        assert "users" in tables
        assert "big_spenders" in tables

    def test_compound_union(self):
        node = parse_ast("""
            SELECT name FROM users
            UNION ALL
            SELECT name FROM products
        """)
        tables = node.tables_referenced()
        assert "users" in tables
        assert "products" in tables

    def test_scalar_subquery_in_select(self):
        node = parse_ast("""
            SELECT u.name,
                   (SELECT COUNT(*) FROM orders o WHERE o.user_id = u.id) AS cnt
            FROM users u
        """)
        tables = node.tables_referenced()
        assert "users" in tables
        assert "orders" in tables

    def test_subquery_in_from(self):
        node = parse_ast("""
            SELECT sub.name
            FROM (SELECT name FROM users) sub
        """)
        tables = node.tables_referenced()
        assert "users" in tables

    def test_no_from(self):
        node = parse_ast("SELECT 1 + 2 AS result")
        assert node.tables_referenced() == []

    def test_deduplication(self):
        node = parse_ast("""
            SELECT u.name, o.total
            FROM users u
            JOIN orders o ON u.id = o.user_id
            WHERE o.total > (SELECT AVG(total) FROM orders)
        """)
        assert node.tables_referenced() == ["users", "orders"]

    def test_compound_deduplication(self):
        node = parse_ast("""
            SELECT name FROM users
            UNION ALL
            SELECT name FROM users
        """)
        assert node.tables_referenced() == ["users"]


# ---------------------------------------------------------------------------
# functions_used()
# ---------------------------------------------------------------------------


class TestFunctionsUsed:
    def test_no_functions(self):
        node = parse_ast("SELECT name FROM users")
        assert node.functions_used() == []

    def test_aggregate_functions(self):
        node = parse_ast("""
            SELECT COUNT(*), SUM(total), AVG(total)
            FROM orders
        """)
        funcs = node.functions_used()
        assert "COUNT" in funcs
        assert "SUM" in funcs
        assert "AVG" in funcs

    def test_nested_functions(self):
        node = parse_ast("""
            SELECT UPPER(SUBSTR(name, 1, 1)), ROUND(AVG(total), 2)
            FROM users u JOIN orders o ON u.id = o.user_id
        """)
        funcs = node.functions_used()
        assert "UPPER" in funcs
        assert "SUBSTR" in funcs
        assert "ROUND" in funcs
        assert "AVG" in funcs

    def test_window_function(self):
        node = parse_ast("""
            SELECT ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY total DESC)
            FROM orders
        """)
        funcs = node.functions_used()
        assert "ROW_NUMBER" in funcs

    def test_compound_functions(self):
        node = parse_ast("""
            SELECT COUNT(*) FROM users
            UNION ALL
            SELECT SUM(total) FROM orders
        """)
        funcs = node.functions_used()
        assert "COUNT" in funcs
        assert "SUM" in funcs

    def test_expression_node_functions_used(self):
        """functions_used on individual expression nodes."""
        node = parse_ast("SELECT UPPER(name), age FROM users")
        assert isinstance(node, Select)
        expr = node.columns[0].expr
        assert expr.functions_used() == ["UPPER"]
        expr2 = node.columns[1].expr
        assert expr2.functions_used() == []

    def test_expression_node_tables_referenced(self):
        """tables_referenced on individual expression nodes."""
        node = parse_ast("""
            SELECT (SELECT COUNT(*) FROM orders WHERE user_id = u.id) FROM users u
        """)
        expr = node.columns[0].expr
        tables = expr.tables_referenced()
        assert "orders" in tables


# ---------------------------------------------------------------------------
# output_columns()
# ---------------------------------------------------------------------------


class TestOutputColumns:
    def test_simple_named_columns(self):
        node = parse_ast("SELECT name, email FROM users")
        cols = node.output_columns()
        assert len(cols) == 2
        assert cols[0] == OutputColumn(table=None, column="name")
        assert cols[1] == OutputColumn(table=None, column="email")

    def test_bare_columns_resolved_with_callback(self):
        """Bare column names resolve to their table when a callback is provided."""
        node = parse_ast("SELECT id, name FROM users")
        cols = node.output_columns(columns_for_table)
        assert cols[0] == OutputColumn(table="users", column="id")
        assert cols[1] == OutputColumn(table="users", column="name")

    def test_bare_columns_ambiguous_left_unresolved(self):
        """Bare column that exists in multiple tables stays unresolved."""
        node = parse_ast("""
            SELECT name FROM users u JOIN products p ON u.id = p.id
        """)
        cols = node.output_columns(columns_for_table)
        assert cols[0] == OutputColumn(table=None, column="name")

    def test_bare_columns_unique_across_join(self):
        """Bare column unique to one table in a JOIN is resolved."""
        node = parse_ast("""
            SELECT email, total FROM users u JOIN orders o ON u.id = o.user_id
        """)
        cols = node.output_columns(columns_for_table)
        assert cols[0] == OutputColumn(table="users", column="email")
        assert cols[1] == OutputColumn(table="orders", column="total")

    def test_qualified_columns(self):
        node = parse_ast("SELECT u.name, u.email FROM users u")
        cols = node.output_columns()
        assert cols[0] == OutputColumn(table="users", column="name")
        assert cols[1] == OutputColumn(table="users", column="email")

    def test_qualified_columns_with_schema(self):
        """Qualified refs resolved via alias map carry the real table name."""
        node = parse_ast("SELECT u.name FROM users u")
        cols = node.output_columns(columns_for_table)
        assert cols[0] == OutputColumn(table="users", column="name")

    def test_aliases(self):
        node = parse_ast("SELECT name AS user_name, COUNT(*) AS cnt FROM users")
        cols = node.output_columns()
        assert cols[0] == OutputColumn(table=None, column="user_name")
        assert cols[1] == OutputColumn(table=None, column="cnt")

    def test_star_expansion(self):
        node = parse_ast("SELECT * FROM users")
        cols = node.output_columns(columns_for_table)
        assert cols == [
            OutputColumn(table="users", column="id"),
            OutputColumn(table="users", column="name"),
            OutputColumn(table="users", column="email"),
            OutputColumn(table="users", column="age"),
            OutputColumn(table="users", column="created_at"),
        ]

    def test_table_star_expansion(self):
        node = parse_ast("SELECT u.* FROM users u")
        cols = node.output_columns(columns_for_table)
        assert cols == [
            OutputColumn(table="users", column="id"),
            OutputColumn(table="users", column="name"),
            OutputColumn(table="users", column="email"),
            OutputColumn(table="users", column="age"),
            OutputColumn(table="users", column="created_at"),
        ]

    def test_star_without_callback(self):
        node = parse_ast("SELECT * FROM users")
        cols = node.output_columns()
        assert cols == [OutputColumn(table=None, column="*")]

    def test_mixed_star_and_named(self):
        node = parse_ast("SELECT u.*, o.total FROM users u JOIN orders o ON 1=1")
        cols = node.output_columns(columns_for_table)
        expected = [
            OutputColumn(table="users", column="id"),
            OutputColumn(table="users", column="name"),
            OutputColumn(table="users", column="email"),
            OutputColumn(table="users", column="age"),
            OutputColumn(table="users", column="created_at"),
            OutputColumn(table="orders", column="total"),
        ]
        assert cols == expected

    def test_cte_output_columns_inferred(self):
        node = parse_ast("""
            WITH totals AS (
                SELECT user_id, SUM(total) AS total_spent FROM orders GROUP BY user_id
            )
            SELECT * FROM totals
        """)
        cols = node.output_columns(columns_for_table)
        assert cols == [
            OutputColumn(table="totals", column="user_id"),
            OutputColumn(table="totals", column="total_spent"),
        ]

    def test_cte_explicit_column_list(self):
        node = parse_ast("""
            WITH summary(uid, spend) AS (
                SELECT user_id, SUM(total) FROM orders GROUP BY user_id
            )
            SELECT * FROM summary
        """)
        cols = node.output_columns(columns_for_table)
        assert cols == [
            OutputColumn(table="summary", column="uid"),
            OutputColumn(table="summary", column="spend"),
        ]

    def test_chained_ctes(self):
        node = parse_ast("""
            WITH order_totals AS (
                SELECT user_id, SUM(total) AS total_spent FROM orders GROUP BY user_id
            ),
            top AS (
                SELECT user_id, total_spent FROM order_totals WHERE total_spent > 5000
            )
            SELECT * FROM top
        """)
        cols = node.output_columns(columns_for_table)
        assert cols == [
            OutputColumn(table="top", column="user_id"),
            OutputColumn(table="top", column="total_spent"),
        ]

    def test_cte_with_star_from_real_table(self):
        node = parse_ast("WITH au AS (SELECT * FROM users) SELECT * FROM au")
        cols = node.output_columns(columns_for_table)
        assert cols == [
            OutputColumn(table="au", column="id"),
            OutputColumn(table="au", column="name"),
            OutputColumn(table="au", column="email"),
            OutputColumn(table="au", column="age"),
            OutputColumn(table="au", column="created_at"),
        ]

    def test_subquery_in_from_output_columns(self):
        node = parse_ast("""
            SELECT * FROM (
                SELECT name, COUNT(*) AS cnt FROM users GROUP BY name
            ) sub
        """)
        cols = node.output_columns(columns_for_table)
        assert cols == [
            OutputColumn(table="sub", column="name"),
            OutputColumn(table="sub", column="cnt"),
        ]

    def test_compound_uses_first_select(self):
        node = parse_ast("""
            SELECT name AS person_name, 'user' AS source FROM users
            UNION ALL
            SELECT name, 'product' FROM products
        """)
        cols = node.output_columns()
        assert cols[0] == OutputColumn(table=None, column="person_name")
        assert cols[1] == OutputColumn(table=None, column="source")

    def test_cte_output_columns_method(self):
        """CTE node's own output_columns method."""
        node = parse_ast("""
            WITH totals AS (
                SELECT user_id, SUM(total) AS spent FROM orders GROUP BY user_id
            )
            SELECT * FROM totals
        """)
        assert isinstance(node, Select)
        cte = node.with_ctes[0]
        cols = cte.output_columns(columns_for_table)
        assert [c.column for c in cols] == ["user_id", "spent"]

    def test_cte_explicit_columns_method(self):
        node = parse_ast("""
            WITH summary(a, b) AS (SELECT 1, 2)
            SELECT * FROM summary
        """)
        cte = node.with_ctes[0]
        cols = cte.output_columns()
        assert [c.column for c in cols] == ["a", "b"]

    def test_subquery_ref_output_columns(self):
        """SubqueryRef node's own output_columns method."""
        node = parse_ast("""
            SELECT sub.x FROM (SELECT name AS x FROM users) sub
        """)
        assert isinstance(node, Select)
        subq_ref = node.from_clause[0]
        cols = subq_ref.output_columns(columns_for_table)
        assert [c.column for c in cols] == ["x"]
