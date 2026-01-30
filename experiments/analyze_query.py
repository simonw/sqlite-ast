"""
Experiment: Can sqlite-ast reliably extract high-level query metadata?

Goals:
- Definitive list of tables used in the query
- Definitive list of table.column columns (resolving SELECT * via columns_for_table callback)
- Which table.column columns appear in ORDER BY (simple column refs only)
- Every function used in the SELECT clause
- Other interesting high-level facts about the query
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from sqlite_ast import parse_ast, ast_nodes as a


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class QueryInfo:
    """High-level metadata extracted from a parsed SQL query."""

    # Tables referenced anywhere in FROM / JOIN / subqueries / CTEs
    tables: list[str] = field(default_factory=list)

    # Fully-qualified columns selected (table.column)
    select_columns: list[str] = field(default_factory=list)

    # Fully-qualified columns in ORDER BY (table.column) – simple refs only
    order_by_columns: list[str] = field(default_factory=list)

    # Functions called in the SELECT expressions
    select_functions: list[str] = field(default_factory=list)

    # Extra observations
    extras: dict = field(default_factory=dict)

    def print(self):
        print("=== Query Analysis ===")
        print(f"  Tables: {self.tables}")
        print(f"  Select columns: {self.select_columns}")
        print(f"  Order-by columns: {self.order_by_columns}")
        print(f"  Select functions: {self.select_functions}")
        for k, v in self.extras.items():
            print(f"  {k}: {v}")
        print()


# ---------------------------------------------------------------------------
# Column resolution helpers (need alias map + schema callback — not on nodes)
# ---------------------------------------------------------------------------


def _resolve_column_ref(
    expr,
    alias_map: dict[str, str],
    columns_for_table: Callable[[str], list[str]] | None,
) -> list[str]:
    """
    Given an expression that is a column reference, return a list of
    resolved 'table.column' strings.

    Handles:
      - Name("col")  → if only one table has that column, resolve it
      - Dot(Name("t"), Name("col"))  → resolve alias
      - Dot(Name("t"), Star())  → expand via columns_for_table
      - Star()  → expand all tables via columns_for_table
    """
    if isinstance(expr, a.Dot):
        left = expr.left
        right = expr.right
        if isinstance(left, a.Name):
            table_name = alias_map.get(left.name, left.name)
            if isinstance(right, a.Name):
                return [f"{table_name}.{right.name}"]
            elif isinstance(right, a.Star):
                if columns_for_table:
                    cols = columns_for_table(table_name)
                    return [f"{table_name}.{c}" for c in cols]
                else:
                    return [f"{table_name}.*"]
    elif isinstance(expr, a.Name):
        col_name = expr.name
        if columns_for_table:
            owners = []
            for real_table in set(alias_map.values()):
                if col_name in columns_for_table(real_table):
                    owners.append(real_table)
            if len(owners) == 1:
                return [f"{owners[0]}.{col_name}"]
        return [col_name]
    elif isinstance(expr, a.Star):
        if columns_for_table:
            result = []
            for real_table in dict.fromkeys(alias_map.values()):
                cols = columns_for_table(real_table)
                result.extend(f"{real_table}.{c}" for c in cols)
            return result
        else:
            return ["*"]
    return []


def _expr_summary(expr) -> str:
    """Short human-readable summary of an expression (for non-column expressions)."""
    if isinstance(expr, a.FunctionCall):
        args = ", ".join(_expr_summary(a_) for a_ in expr.args)
        return f"{expr.name}({args})"
    elif isinstance(expr, a.Name):
        return expr.name
    elif isinstance(expr, a.Dot):
        return f"{_expr_summary(expr.left)}.{_expr_summary(expr.right)}"
    elif isinstance(expr, a.Star):
        return "*"
    elif isinstance(expr, a.IntegerLiteral):
        return str(expr.value)
    elif isinstance(expr, a.StringLiteral):
        return repr(expr.value)
    elif isinstance(expr, a.BinaryOp):
        return f"({_expr_summary(expr.left)} {expr.op} {_expr_summary(expr.right)})"
    elif isinstance(expr, a.UnaryOp):
        return f"({expr.op} {_expr_summary(expr.operand)})"
    elif isinstance(expr, a.Cast):
        return f"CAST({_expr_summary(expr.expr)} AS {expr.as_type})"
    elif isinstance(expr, a.Case):
        return "CASE..."
    elif isinstance(expr, a.Subquery):
        return "(subquery)"
    elif isinstance(expr, a.NullLiteral):
        return "NULL"
    elif isinstance(expr, a.FloatLiteral):
        return expr.value
    elif isinstance(expr, a.Collate):
        return f"{_expr_summary(expr.expr)} COLLATE {expr.collation}"
    else:
        return f"<{type(expr).__name__}>"


# ---------------------------------------------------------------------------
# Main analysis entry point
# ---------------------------------------------------------------------------


def analyze_query(
    sql: str,
    columns_for_table: Callable[[str], list[str]] | None = None,
) -> QueryInfo:
    """
    Parse a SQL SELECT statement and extract high-level metadata.

    Args:
        sql: A SQL SELECT statement.
        columns_for_table: Optional callback that returns list of column
            names for a given table name. Used to resolve `SELECT *` and
            `SELECT t.*` as well as bare column names.
    """
    node = parse_ast(sql)
    info = QueryInfo()

    # --- unwrap: work with the "outer" select ---
    outer = node
    if isinstance(node, a.Compound):
        outer = node.body[0].select

    # Build augmented column lookup that knows CTE + subquery schemas
    aug_columns = outer._augmented_columns_for_table(columns_for_table)

    # 1. Tables — use the node method
    raw_tables = node.tables_referenced()
    info.tables = list(dict.fromkeys(raw_tables))  # dedupe, preserve order

    # 2. Alias map (from outermost FROM)
    alias_map = outer._alias_map()

    # 3. Select columns (resolve stars, qualified refs, bare names)
    for col in outer.columns:
        resolved = _resolve_column_ref(col.expr, alias_map, aug_columns)
        if resolved:
            info.select_columns.extend(resolved)
        else:
            if col.alias:
                info.select_columns.append(col.alias)
            else:
                info.select_columns.append(_expr_summary(col.expr))

    # 4. Order-by columns
    order_by = node.order_by if hasattr(node, "order_by") else None
    if order_by:
        for item in order_by:
            resolved = _resolve_column_ref(item.expr, alias_map, aug_columns)
            if resolved:
                direction = item.direction
                for r in resolved:
                    info.order_by_columns.append(f"{r} {direction}")
            elif isinstance(item.expr, a.IntegerLiteral):
                idx = item.expr.value - 1
                if 0 <= idx < len(info.select_columns):
                    info.order_by_columns.append(
                        f"{info.select_columns[idx]} {item.direction} (positional)"
                    )
                else:
                    info.order_by_columns.append(
                        f"position({item.expr.value}) {item.direction}"
                    )
            else:
                info.order_by_columns.append(
                    f"<expr: {_expr_summary(item.expr)}> {item.direction}"
                )

    # 5. Functions in SELECT — use the node method
    info.select_functions = list(dict.fromkeys(node.functions_used()))

    # 6. Extras
    info.extras["has_where"] = outer.where is not None
    # Expose inferred CTE/subquery schemas if any were discovered
    if outer.with_ctes:
        cte_schemas = {}
        for cte in outer.with_ctes:
            cte_schemas[cte.name] = aug_columns(cte.name)
        if cte_schemas:
            info.extras["cte_schemas"] = cte_schemas
    if outer.from_clause:
        subquery_schemas = {}
        for item in outer.from_clause:
            if isinstance(item, a.SubqueryRef) and item.alias:
                subquery_schemas[item.alias] = aug_columns(item.alias)
        if subquery_schemas:
            info.extras["subquery_schemas"] = subquery_schemas
    info.extras["has_group_by"] = outer.group_by is not None
    info.extras["has_having"] = outer.having is not None
    info.extras["has_limit"] = outer._has_limit
    info.extras["has_order_by"] = order_by is not None
    info.extras["is_distinct"] = outer.distinct
    info.extras["is_compound"] = isinstance(node, a.Compound)
    if isinstance(node, a.Compound):
        ops = [p.operator for p in node.body if p.operator]
        info.extras["compound_operators"] = ops
    if outer.with_ctes:
        info.extras["cte_names"] = [c.name for c in outer.with_ctes]
    # Count joins
    if outer.from_clause:
        joins = [
            f for f in outer.from_clause
            if isinstance(f, (a.TableRef, a.SubqueryRef)) and f.join_type is not None
        ]
        if joins:
            info.extras["joins"] = [
                {"type": j.join_type, "table": j.name if isinstance(j, a.TableRef) else "(subquery)"}
                for j in joins
            ]
    # Subqueries in WHERE
    if outer.where:
        sub_tables = outer.where.tables_referenced()
        if sub_tables:
            info.extras["where_references_tables"] = list(dict.fromkeys(sub_tables))

    return info


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------


def main():
    print("=" * 72)
    print("EXPERIMENT: SQL Query Analysis using sqlite-ast")
    print("=" * 72)
    print()

    # Simulated schema for star expansion
    SCHEMA = {
        "users": ["id", "name", "email", "age", "created_at"],
        "orders": ["id", "user_id", "product_id", "quantity", "total", "created_at"],
        "products": ["id", "name", "price", "category", "description"],
        "categories": ["id", "name", "parent_id"],
        "reviews": ["id", "user_id", "product_id", "rating", "comment", "created_at"],
    }

    def columns_for_table(table: str) -> list[str]:
        return SCHEMA.get(table, [])

    # ------------------------------------------------------------------
    tests = []

    # --- Test 1: Simple select with explicit columns ---
    tests.append((
        "1. Simple SELECT with explicit columns",
        "SELECT name, email FROM users WHERE age > 18 ORDER BY name ASC",
    ))

    # --- Test 2: SELECT * expansion ---
    tests.append((
        "2. SELECT * expansion",
        "SELECT * FROM users ORDER BY created_at DESC",
    ))

    # --- Test 3: SELECT table.* ---
    tests.append((
        "3. SELECT table.* with JOIN",
        """
        SELECT u.*, o.total
        FROM users u
        JOIN orders o ON u.id = o.user_id
        ORDER BY o.total DESC
        """,
    ))

    # --- Test 4: Multi-table JOIN with mixed references ---
    tests.append((
        "4. Multi-table JOIN with mixed column refs",
        """
        SELECT u.name, p.name, o.quantity, o.total
        FROM orders o
        JOIN users u ON u.id = o.user_id
        JOIN products p ON p.id = o.product_id
        WHERE o.total > 100
        ORDER BY u.name, o.total DESC
        """,
    ))

    # --- Test 5: Functions in SELECT ---
    tests.append((
        "5. Aggregate functions",
        """
        SELECT
            u.name,
            COUNT(*) AS order_count,
            SUM(o.total) AS total_spent,
            AVG(o.total) AS avg_order,
            MAX(o.created_at) AS last_order
        FROM users u
        JOIN orders o ON u.id = o.user_id
        GROUP BY u.name
        HAVING COUNT(*) > 5
        ORDER BY total_spent DESC
        """,
    ))

    # --- Test 6: Subquery in WHERE ---
    tests.append((
        "6. Subquery in WHERE",
        """
        SELECT name, email
        FROM users
        WHERE id IN (
            SELECT user_id FROM orders WHERE total > 500
        )
        ORDER BY name
        """,
    ))

    # --- Test 7: CTE (WITH clause) ---
    tests.append((
        "7. CTE (WITH clause)",
        """
        WITH big_spenders AS (
            SELECT user_id, SUM(total) AS total_spent
            FROM orders
            GROUP BY user_id
            HAVING SUM(total) > 1000
        )
        SELECT u.name, bs.total_spent
        FROM users u
        JOIN big_spenders bs ON u.id = bs.user_id
        ORDER BY bs.total_spent DESC
        """,
    ))

    # --- Test 8: Window functions ---
    tests.append((
        "8. Window functions",
        """
        SELECT
            u.name,
            o.total,
            ROW_NUMBER() OVER (PARTITION BY u.id ORDER BY o.total DESC) AS rn,
            SUM(o.total) OVER (PARTITION BY u.id) AS user_total
        FROM users u
        JOIN orders o ON u.id = o.user_id
        ORDER BY u.name, rn
        """,
    ))

    # --- Test 9: Nested functions ---
    tests.append((
        "9. Nested function calls",
        """
        SELECT
            UPPER(SUBSTR(u.name, 1, 1)) AS initial,
            COALESCE(u.email, 'unknown') AS email,
            ROUND(AVG(o.total), 2) AS avg_total,
            CAST(COUNT(*) AS TEXT) AS cnt
        FROM users u
        LEFT JOIN orders o ON u.id = o.user_id
        GROUP BY u.name, u.email
        ORDER BY avg_total DESC
        """,
    ))

    # --- Test 10: Compound query (UNION) ---
    tests.append((
        "10. UNION query",
        """
        SELECT name, 'user' AS source FROM users
        UNION ALL
        SELECT name, 'product' AS source FROM products
        ORDER BY name
        """,
    ))

    # --- Test 11: CASE expression ---
    tests.append((
        "11. CASE expression in SELECT",
        """
        SELECT
            u.name,
            CASE
                WHEN o.total > 1000 THEN 'high'
                WHEN o.total > 100 THEN 'medium'
                ELSE 'low'
            END AS spend_tier,
            o.total
        FROM users u
        JOIN orders o ON u.id = o.user_id
        ORDER BY o.total DESC
        """,
    ))

    # --- Test 12: Positional ORDER BY ---
    tests.append((
        "12. Positional ORDER BY",
        """
        SELECT u.name, COUNT(*) AS cnt
        FROM users u
        JOIN orders o ON u.id = o.user_id
        GROUP BY u.name
        ORDER BY 2 DESC
        """,
    ))

    # --- Test 13: Complex real-world query ---
    tests.append((
        "13. Complex real-world-ish query",
        """
        WITH active_users AS (
            SELECT DISTINCT user_id
            FROM orders
            WHERE created_at > '2024-01-01'
        ),
        user_stats AS (
            SELECT
                o.user_id,
                COUNT(*) AS order_count,
                SUM(o.total) AS total_spent,
                AVG(o.total) AS avg_order
            FROM orders o
            WHERE o.user_id IN (SELECT user_id FROM active_users)
            GROUP BY o.user_id
        )
        SELECT
            u.name,
            u.email,
            us.order_count,
            us.total_spent,
            ROUND(us.avg_order, 2) AS avg_order,
            CASE
                WHEN us.total_spent > 10000 THEN 'platinum'
                WHEN us.total_spent > 5000 THEN 'gold'
                WHEN us.total_spent > 1000 THEN 'silver'
                ELSE 'bronze'
            END AS tier
        FROM users u
        JOIN user_stats us ON u.id = us.user_id
        ORDER BY us.total_spent DESC
        LIMIT 50
        """,
    ))

    # --- Test 14: Self-join ---
    tests.append((
        "14. Self-join (categories with parent)",
        """
        SELECT c.name AS category, p.name AS parent_category
        FROM categories c
        LEFT JOIN categories p ON c.parent_id = p.id
        ORDER BY p.name, c.name
        """,
    ))

    # --- Test 15: EXISTS subquery ---
    tests.append((
        "15. EXISTS subquery",
        """
        SELECT u.name, u.email
        FROM users u
        WHERE EXISTS (
            SELECT 1 FROM reviews r
            WHERE r.user_id = u.id AND r.rating = 5
        )
        ORDER BY u.name
        """,
    ))

    # --- Test 16: Bare column names without table prefix, single table ---
    tests.append((
        "16. Bare column names (single table, resolvable)",
        """
        SELECT name, email, age
        FROM users
        WHERE age > 21
        ORDER BY name
        """,
    ))

    # --- Test 17: Ambiguous bare column name (name exists in multiple tables) ---
    tests.append((
        "17. Ambiguous bare column (exists in users AND products)",
        """
        SELECT name
        FROM users u
        JOIN products p ON u.id = p.id
        """,
    ))

    # --- Test 18: ORDER BY alias ---
    tests.append((
        "18. ORDER BY using a SELECT alias",
        """
        SELECT u.name, SUM(o.total) AS total_spent
        FROM users u
        JOIN orders o ON u.id = o.user_id
        GROUP BY u.name
        ORDER BY total_spent DESC
        """,
    ))

    # --- Test 19: Subquery in FROM ---
    tests.append((
        "19. Subquery in FROM clause",
        """
        SELECT sub.user_name, sub.order_count
        FROM (
            SELECT u.name AS user_name, COUNT(*) AS order_count
            FROM users u
            JOIN orders o ON u.id = o.user_id
            GROUP BY u.name
        ) sub
        ORDER BY sub.order_count DESC
        """,
    ))

    # --- Test 20: LIKE (parsed as function by SQLite) ---
    tests.append((
        "20. LIKE operator (SQLite parses as function call)",
        """
        SELECT name, email
        FROM users
        WHERE name LIKE '%smith%'
        ORDER BY name
        """,
    ))

    # --- Test 21: Multiple aggregates + expressions in ORDER BY ---
    tests.append((
        "21. Expression in ORDER BY (non-simple)",
        """
        SELECT u.name, o.total
        FROM users u
        JOIN orders o ON u.id = o.user_id
        ORDER BY o.total * o.quantity DESC
        """,
    ))

    # --- Test 22: No FROM clause ---
    tests.append((
        "22. No FROM clause (computed values only)",
        """
        SELECT 1 + 2 AS result, 'hello' AS greeting, UPPER('world') AS upper_world
        """,
    ))

    # --- Test 23: DISTINCT ---
    tests.append((
        "23. SELECT DISTINCT",
        """
        SELECT DISTINCT u.name, u.email
        FROM users u
        JOIN orders o ON u.id = o.user_id
        WHERE o.total > 100
        """,
    ))

    # --- Test 24: Multiple subqueries in different positions ---
    tests.append((
        "24. Subqueries in SELECT, WHERE, and FROM",
        """
        SELECT
            u.name,
            (SELECT COUNT(*) FROM orders o2 WHERE o2.user_id = u.id) AS order_count
        FROM users u
        WHERE u.id IN (SELECT DISTINCT user_id FROM reviews)
        ORDER BY u.name
        """,
    ))

    # --- Test 25: CTE bare column resolution ---
    tests.append((
        "25. CTE columns resolved (bare refs from CTE disambiguated)",
        """
        WITH user_totals AS (
            SELECT u.id AS user_id, u.name, SUM(o.total) AS total_spent
            FROM users u
            JOIN orders o ON u.id = o.user_id
            GROUP BY u.id, u.name
        )
        SELECT user_id, name, total_spent
        FROM user_totals
        WHERE total_spent > 1000
        ORDER BY total_spent DESC
        """,
    ))

    # --- Test 26: CTE with SELECT * from CTE ---
    tests.append((
        "26. SELECT * from a CTE",
        """
        WITH recent_orders AS (
            SELECT o.id, o.user_id, o.total, o.created_at
            FROM orders o
            WHERE o.created_at > '2024-01-01'
        )
        SELECT *
        FROM recent_orders
        ORDER BY created_at DESC
        """,
    ))

    # --- Test 27: Subquery in FROM with column resolution ---
    tests.append((
        "27. Subquery-in-FROM columns resolved",
        """
        SELECT sub.user_name, sub.order_count
        FROM (
            SELECT u.name AS user_name, COUNT(*) AS order_count
            FROM users u
            JOIN orders o ON u.id = o.user_id
            GROUP BY u.name
        ) sub
        ORDER BY sub.order_count DESC
        """,
    ))

    # --- Test 28: Subquery in FROM with SELECT * ---
    tests.append((
        "28. SELECT * from subquery-in-FROM",
        """
        SELECT *
        FROM (
            SELECT u.name, u.email, COUNT(*) AS cnt
            FROM users u
            JOIN orders o ON u.id = o.user_id
            GROUP BY u.name, u.email
        ) stats
        ORDER BY cnt DESC
        """,
    ))

    # --- Test 29: Chained CTEs (CTE2 references CTE1) ---
    tests.append((
        "29. Chained CTEs (CTE2 reads from CTE1)",
        """
        WITH order_totals AS (
            SELECT user_id, SUM(total) AS total_spent
            FROM orders
            GROUP BY user_id
        ),
        top_spenders AS (
            SELECT user_id, total_spent
            FROM order_totals
            WHERE total_spent > 5000
        )
        SELECT u.name, ts.total_spent
        FROM users u
        JOIN top_spenders ts ON u.id = ts.user_id
        ORDER BY ts.total_spent DESC
        """,
    ))

    # --- Test 30: CTE with explicit column list ---
    tests.append((
        "30. CTE with explicit column list",
        """
        WITH summary(uid, spend) AS (
            SELECT user_id, SUM(total)
            FROM orders
            GROUP BY user_id
        )
        SELECT u.name, s.spend
        FROM users u
        JOIN summary s ON u.id = s.uid
        ORDER BY s.spend DESC
        """,
    ))

    # --- Test 31: CTE with SELECT * expansion inside the CTE ---
    tests.append((
        "31. CTE that uses SELECT * from a real table",
        """
        WITH all_users AS (
            SELECT * FROM users
        )
        SELECT name, email
        FROM all_users
        ORDER BY name
        """,
    ))

    # --- Test 32: Subquery-in-FROM joined with real table ---
    tests.append((
        "32. Subquery-in-FROM joined with real table",
        """
        SELECT u.name, sub.total_orders, sub.total_spent
        FROM users u
        JOIN (
            SELECT user_id, COUNT(*) AS total_orders, SUM(total) AS total_spent
            FROM orders
            GROUP BY user_id
        ) sub ON u.id = sub.user_id
        ORDER BY sub.total_spent DESC
        """,
    ))

    # --- Test 33: CTE + subquery-in-FROM combined ---
    tests.append((
        "33. CTE + subquery-in-FROM combined",
        """
        WITH active AS (
            SELECT DISTINCT user_id
            FROM orders
            WHERE created_at > '2024-01-01'
        )
        SELECT u.name, stats.order_count
        FROM users u
        JOIN (
            SELECT a.user_id, COUNT(*) AS order_count
            FROM active a
            JOIN orders o ON a.user_id = o.user_id
            GROUP BY a.user_id
        ) stats ON u.id = stats.user_id
        ORDER BY stats.order_count DESC
        """,
    ))

    # ------------------------------------------------------------------
    # Run tests
    # ------------------------------------------------------------------

    passed = 0
    failed = 0
    for title, sql in tests:
        print("-" * 72)
        print(f"TEST: {title}")
        print(f"  SQL: {' '.join(sql.split())}")
        try:
            info = analyze_query(sql, columns_for_table)
            info.print()
            passed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            print()

    print("=" * 72)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print("=" * 72)

    # ------------------------------------------------------------------
    # Findings summary
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("FINDINGS SUMMARY")
    print("=" * 72)
    print("""
WHAT WORKS RELIABLY
===================

1. TABLE EXTRACTION (high confidence)
   - Tables in FROM / JOIN are reliably extracted via TableRef nodes.
   - Tables in subqueries (WHERE IN, EXISTS, scalar subqueries) are found
     by recursing into Subquery/Exists/InList expression nodes.
   - Tables inside CTEs are found by recursing into CTE.select.
   - CTE names themselves appear as TableRef when referenced in FROM.
   - The library distinguishes CTE names from real tables (cte_names extra).

2. SELECT COLUMN RESOLUTION (high confidence, with caveats)
   - Qualified refs (u.name) → resolved via alias map → "users.name". Works.
   - SELECT * → expanded via columns_for_table callback → all columns. Works.
   - SELECT t.* → expanded for that alias's table. Works.
   - Bare column names (single table) → resolved when unambiguous. Works.
   - Bare column names (multi-table, ambiguous) → falls back to bare name
     because we can't know which table owns it without schema. Correct behavior.
   - Expression columns (functions, CASE, etc.) → reported by alias if present,
     or by a short expression summary. Good enough for metadata.

3. CTE + SUBQUERY-IN-FROM COLUMN INFERENCE (high confidence)
   CTE and subquery-in-FROM output columns are inferred by analyzing their
   SELECT lists. This enables:
   - Bare column refs from CTEs resolved (Test 25):
     `SELECT user_id FROM user_totals` → "user_totals.user_id"
   - SELECT * from CTEs expanded (Test 26):
     `SELECT * FROM recent_orders` → all inferred CTE columns
   - Subquery-in-FROM with SELECT * (Test 28):
     `SELECT * FROM (...) stats` → "stats.name", "stats.email", "stats.cnt"
   - Chained CTEs (Test 29): CTE2 referencing CTE1 works because CTEs are
     processed in order, each augmenting the lookup for later ones.
   - Explicit CTE column lists (Test 30):
     `WITH summary(uid, spend) AS (...)` → uses the explicit names.
   - CTE using SELECT * from a real table (Test 31):
     `With all_users AS (SELECT * FROM users)` → inherits users' columns.
   - Combined CTE + subquery-in-FROM (Test 33) works.

   Now implemented as methods on the AST nodes:
     - Select.output_columns(columns_for_table) → column names produced
     - CTE.output_columns(columns_for_table) → explicit list or inferred
     - SubqueryRef.output_columns(columns_for_table) → delegates to select
     - Compound.output_columns(columns_for_table) → first SELECT's columns

4. ORDER BY COLUMNS (high confidence for simple cases)
   - Qualified refs (o.total) → resolved via alias map. Works.
   - Bare column names → resolved when unambiguous. Works.
   - Positional ORDER BY (ORDER BY 2) → mapped to the Nth select column. Works.
   - ORDER BY alias name → reported as the alias string. The alias IS the
     column identity at that point, so this is correct.
   - Expression ORDER BY (ORDER BY o.total * o.quantity) → reported as
     "<expr: ...>" with a summary. Correctly identified as non-simple.

5. FUNCTION EXTRACTION (high confidence)
   - All FunctionCall nodes in the SELECT are collected, including nested ones.
   - UPPER(SUBSTR(...)) correctly reports both UPPER and SUBSTR.
   - Window functions (ROW_NUMBER, SUM OVER) are correctly identified.
   - count(*) is correctly detected (SQLite produces empty args list for it).
   - NOTE: SQLite parses LIKE/GLOB/MATCH as function calls with reversed args.
     So `name LIKE '%x%'` shows up as a LIKE function. This is a feature of
     the parser matching SQLite's own behavior, not a bug.

   Now implemented as methods on the AST nodes:
     - Every expression node has .functions_used() → list[str]
     - Select.functions_used() → functions from SELECT column expressions
     - Compound.functions_used() → functions across all branches

6. HIGH-LEVEL EXTRAS (reliable)
   - has_where, has_group_by, has_having, has_limit, has_order_by, is_distinct
   - is_compound + compound_operators (UNION, UNION ALL, etc.)
   - cte_names + cte_schemas (inferred column lists per CTE)
   - subquery_schemas (inferred column lists per subquery-in-FROM)
   - joins with type (JOIN, LEFT JOIN, etc.) and table name
   - where_references_tables (tables found in WHERE subqueries)

   Now implemented as methods on the AST nodes:
     - Every expression node has .tables_referenced() → list[str]
     - Select/Compound.tables_referenced() → all tables recursively
     - TableRef.tables_referenced() → [self.name]
     - SubqueryRef/CTE/CompoundPart.tables_referenced() → delegates to select

KNOWN LIMITATIONS / EDGE CASES
================================

1. SELF-JOIN ALIAS COLLAPSE (Test 14)
   The alias map resolves both c→categories and p→categories, so c.name
   and p.name both become "categories.name". The two columns are truly from
   the same underlying table, but the alias distinction is lost. To preserve
   it, you'd need to keep the alias as part of the column identity
   (e.g., "c.name" vs "p.name" or "categories[c].name" vs "categories[p].name").
   This is inherent to resolving aliases to real table names.

2. COMPOUND QUERIES (UNION)
   For UNION/INTERSECT/EXCEPT, we analyze only the first SELECT for columns.
   The column names come from the first branch, which is SQL-standard behavior.
   Tables are collected from ALL branches, which is correct.

3. EXPRESSION COLUMNS WITHOUT ALIASES
   When a CTE or subquery SELECT list contains an expression without an alias
   (e.g., `SELECT 1+2, COUNT(*)`), we use _expr_column_name() as the column
   name. SQLite internally uses the expression text, which is close but not
   guaranteed identical. In practice, such columns almost always have aliases
   in real-world queries.

4. EXPRESSION COLUMNS IN ORDER BY
   ORDER BY with expressions (e.g., o.total * o.quantity) are correctly
   flagged as non-simple. No attempt to decompose the expression into
   constituent column refs, which would be possible but requires more work.

OVERALL ASSESSMENT
==================
sqlite-ast provides a reliable, complete AST that makes all of the above
extractions straightforward. The AST structure maps directly to the query
structure, with clean dataclass nodes that are easy to walk recursively.

Analysis methods are now part of the AST node API:
  - node.tables_referenced()       → all table names in the subtree
  - node.functions_used()          → all function names in the subtree
  - node.output_columns(callback)  → column names this query produces

These work on Select, Compound, CTE, SubqueryRef, and all expression nodes.
""")



if __name__ == "__main__":
    main()
