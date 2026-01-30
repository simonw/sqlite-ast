"""AST node dataclasses for SQLite SELECT statements."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable


# --- Base class for expression nodes ---


class ExprBase:
    """Base for expression nodes, providing default no-op analysis methods."""

    def tables_referenced(self) -> list[str]:
        return []

    def functions_used(self) -> list[str]:
        return []


# --- Output column descriptor ---


@dataclass
class OutputColumn:
    """A column produced by a SELECT, with optional source table."""
    table: str | None
    column: str

    def __str__(self) -> str:
        if self.table:
            return f"{self.table}.{self.column}"
        return self.column

    def to_dict(self) -> dict:
        return {"table": self.table, "column": self.column}


# --- Expression leaf nodes ---

@dataclass
class IntegerLiteral(ExprBase):
    value: int

    def to_dict(self) -> dict:
        return {"type": "integer", "value": self.value}


@dataclass
class FloatLiteral(ExprBase):
    value: str  # original text representation

    def to_dict(self) -> dict:
        return {"type": "float", "value": self.value}


@dataclass
class StringLiteral(ExprBase):
    value: str

    def to_dict(self) -> dict:
        return {"type": "string", "value": self.value}


@dataclass
class BlobLiteral(ExprBase):
    value: str  # full X'...' text

    def to_dict(self) -> dict:
        return {"type": "blob", "value": self.value}


@dataclass
class NullLiteral(ExprBase):
    def to_dict(self) -> dict:
        return {"type": "null"}


@dataclass
class Name(ExprBase):
    name: str

    def to_dict(self) -> dict:
        return {"type": "name", "name": self.name}


@dataclass
class Star(ExprBase):
    def to_dict(self) -> dict:
        return {"type": "star"}


@dataclass
class Parameter(ExprBase):
    name: str

    def to_dict(self) -> dict:
        return {"type": "parameter", "name": self.name}


@dataclass
class Unknown(ExprBase):
    """Placeholder node (used for JOIN USING internal representation)."""
    op: int

    def to_dict(self) -> dict:
        return {"type": "unknown", "op": self.op}


# --- Expression compound nodes ---

@dataclass
class UnaryOp(ExprBase):
    op: str
    operand: Any  # expression node

    def to_dict(self) -> dict:
        return {"type": "unary", "op": self.op, "operand": self.operand.to_dict()}

    def tables_referenced(self) -> list[str]:
        return self.operand.tables_referenced()

    def functions_used(self) -> list[str]:
        return self.operand.functions_used()


@dataclass
class BinaryOp(ExprBase):
    op: str
    left: Any
    right: Any

    def to_dict(self) -> dict:
        return {
            "type": "binary",
            "op": self.op,
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
        }

    def tables_referenced(self) -> list[str]:
        return self.left.tables_referenced() + self.right.tables_referenced()

    def functions_used(self) -> list[str]:
        return self.left.functions_used() + self.right.functions_used()


@dataclass
class Dot(ExprBase):
    left: Any
    right: Any

    def to_dict(self) -> dict:
        return {
            "type": "dot",
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
        }

    def tables_referenced(self) -> list[str]:
        return self.left.tables_referenced() + self.right.tables_referenced()

    def functions_used(self) -> list[str]:
        return self.left.functions_used() + self.right.functions_used()


@dataclass
class FunctionCall(ExprBase):
    name: str
    args: list
    distinct: bool = False
    over: WindowSpec | None = None

    def to_dict(self) -> dict:
        d: dict = {
            "type": "function",
            "name": self.name,
            "args": [a.to_dict() for a in self.args],
            "distinct": self.distinct,
        }
        if self.over is not None:
            d["over"] = self.over.to_dict()
        return d

    def tables_referenced(self) -> list[str]:
        tables: list[str] = []
        for arg in self.args:
            tables.extend(arg.tables_referenced())
        if self.over is not None:
            tables.extend(self.over.tables_referenced())
        return tables

    def functions_used(self) -> list[str]:
        funcs = [self.name]
        for arg in self.args:
            funcs.extend(arg.functions_used())
        if self.over is not None:
            funcs.extend(self.over.functions_used())
        return funcs


@dataclass
class Cast(ExprBase):
    expr: Any
    as_type: str

    def to_dict(self) -> dict:
        return {"type": "cast", "expr": self.expr.to_dict(), "as": self.as_type}

    def tables_referenced(self) -> list[str]:
        return self.expr.tables_referenced()

    def functions_used(self) -> list[str]:
        return self.expr.functions_used()


@dataclass
class Case(ExprBase):
    operand: Any | None
    when_clauses: list
    else_expr: Any | None

    def to_dict(self) -> dict:
        return {
            "type": "case",
            "operand": self.operand.to_dict() if self.operand else None,
            "when_clauses": [
                {
                    "when": w.to_dict(),
                    "then": t.to_dict(),
                }
                for w, t in self.when_clauses
            ],
            "else": self.else_expr.to_dict() if self.else_expr else None,
        }

    def tables_referenced(self) -> list[str]:
        tables: list[str] = []
        if self.operand:
            tables.extend(self.operand.tables_referenced())
        for w, t in self.when_clauses:
            tables.extend(w.tables_referenced())
            tables.extend(t.tables_referenced())
        if self.else_expr:
            tables.extend(self.else_expr.tables_referenced())
        return tables

    def functions_used(self) -> list[str]:
        funcs: list[str] = []
        if self.operand:
            funcs.extend(self.operand.functions_used())
        for w, t in self.when_clauses:
            funcs.extend(w.functions_used())
            funcs.extend(t.functions_used())
        if self.else_expr:
            funcs.extend(self.else_expr.functions_used())
        return funcs


@dataclass
class Between(ExprBase):
    expr: Any
    low: Any
    high: Any

    def to_dict(self) -> dict:
        return {
            "type": "between",
            "expr": self.expr.to_dict(),
            "low": self.low.to_dict(),
            "high": self.high.to_dict(),
        }

    def tables_referenced(self) -> list[str]:
        return (
            self.expr.tables_referenced()
            + self.low.tables_referenced()
            + self.high.tables_referenced()
        )

    def functions_used(self) -> list[str]:
        return (
            self.expr.functions_used()
            + self.low.functions_used()
            + self.high.functions_used()
        )


@dataclass
class InList(ExprBase):
    expr: Any
    values: list | None = None
    select: Select | Compound | None = None

    def to_dict(self) -> dict:
        d: dict = {"type": "in", "expr": self.expr.to_dict()}
        if self.select is not None:
            d["select"] = self.select.to_dict()
        else:
            d["values"] = [v.to_dict() for v in (self.values or [])]
        return d

    def tables_referenced(self) -> list[str]:
        tables = self.expr.tables_referenced()
        if self.select is not None:
            tables.extend(self.select.tables_referenced())
        if self.values:
            for v in self.values:
                tables.extend(v.tables_referenced())
        return tables

    def functions_used(self) -> list[str]:
        funcs = self.expr.functions_used()
        if self.select is not None:
            funcs.extend(self.select.functions_used())
        if self.values:
            for v in self.values:
                funcs.extend(v.functions_used())
        return funcs


@dataclass
class Exists(ExprBase):
    select: Any

    def to_dict(self) -> dict:
        return {"type": "exists", "select": self.select.to_dict()}

    def tables_referenced(self) -> list[str]:
        return self.select.tables_referenced()

    def functions_used(self) -> list[str]:
        return self.select.functions_used()


@dataclass
class Subquery(ExprBase):
    select: Any

    def to_dict(self) -> dict:
        return {"type": "subquery", "select": self.select.to_dict()}

    def tables_referenced(self) -> list[str]:
        return self.select.tables_referenced()

    def functions_used(self) -> list[str]:
        return self.select.functions_used()


@dataclass
class Collate(ExprBase):
    expr: Any
    collation: str

    def to_dict(self) -> dict:
        return {
            "type": "collate",
            "expr": self.expr.to_dict(),
            "collation": self.collation,
        }

    def tables_referenced(self) -> list[str]:
        return self.expr.tables_referenced()

    def functions_used(self) -> list[str]:
        return self.expr.functions_used()


@dataclass
class IsNull(ExprBase):
    operand: Any

    def to_dict(self) -> dict:
        return {"type": "isnull", "operand": self.operand.to_dict()}

    def tables_referenced(self) -> list[str]:
        return self.operand.tables_referenced()

    def functions_used(self) -> list[str]:
        return self.operand.functions_used()


@dataclass
class NotNull(ExprBase):
    operand: Any

    def to_dict(self) -> dict:
        return {"type": "notnull", "operand": self.operand.to_dict()}

    def tables_referenced(self) -> list[str]:
        return self.operand.tables_referenced()

    def functions_used(self) -> list[str]:
        return self.operand.functions_used()


# --- Helpers ---

def _expr_column_name(expr: Any) -> str:
    """Fallback column name for an expression without an alias."""
    if isinstance(expr, Name):
        return expr.name
    if isinstance(expr, FunctionCall):
        return f"{expr.name}(...)"
    if isinstance(expr, Star):
        return "*"
    return "<expr>"


# --- SELECT statement nodes ---

@dataclass
class ResultColumn:
    expr: Any
    alias: str | None

    def to_dict(self) -> dict:
        return {"expr": self.expr.to_dict(), "alias": self.alias}


@dataclass
class TableRef:
    name: str
    alias: str | None = None
    join_type: str | None = None
    on: Any | None = None
    using: list[str] | None = None
    schema: str | None = None

    def to_dict(self) -> dict:
        d: dict = {"type": "table", "name": self.name}
        if self.schema is not None:
            d["schema"] = self.schema
        d["alias"] = self.alias
        d["join_type"] = self.join_type
        if self.on is not None:
            d["on"] = self.on.to_dict()
        if self.using is not None:
            d["using"] = self.using
        return d

    def tables_referenced(self) -> list[str]:
        return [self.name]


@dataclass
class SubqueryRef:
    select: Any
    alias: str | None = None
    join_type: str | None = None
    on: Any | None = None

    def to_dict(self) -> dict:
        d: dict = {
            "type": "subquery",
            "select": self.select.to_dict(),
            "alias": self.alias,
            "join_type": self.join_type,
        }
        if self.on is not None:
            d["on"] = self.on.to_dict()
        return d

    def tables_referenced(self) -> list[str]:
        return self.select.tables_referenced()

    def output_columns(
        self,
        columns_for_table: Callable[[str], list[str]] | None = None,
    ) -> list[OutputColumn]:
        return self.select.output_columns(columns_for_table)


@dataclass
class OrderByItem:
    expr: Any
    direction: str = "ASC"
    nulls: str | None = None

    def to_dict(self) -> dict:
        d: dict = {"expr": self.expr.to_dict(), "direction": self.direction}
        if self.nulls is not None:
            d["nulls"] = self.nulls
        return d


@dataclass
class CTE:
    name: str
    select: Any
    columns: list[str] | None = None
    materialized: str | None = None

    def to_dict(self) -> dict:
        d: dict = {"name": self.name}
        if self.columns is not None:
            d["columns"] = self.columns
        if self.materialized is not None:
            d["materialized"] = self.materialized
        d["select"] = self.select.to_dict()
        return d

    def tables_referenced(self) -> list[str]:
        return self.select.tables_referenced()

    def output_columns(
        self,
        columns_for_table: Callable[[str], list[str]] | None = None,
    ) -> list[OutputColumn]:
        if self.columns:
            return [OutputColumn(table=None, column=c) for c in self.columns]
        return self.select.output_columns(columns_for_table)


@dataclass
class WindowDef:
    name: str
    base: str | None = None
    partition_by: list | None = None
    order_by: list[OrderByItem] | None = None
    frame: FrameSpec | None = None

    def to_dict(self) -> dict:
        d: dict = {"name": self.name, "base": self.base}
        if self.partition_by is not None:
            d["partition_by"] = [e.to_dict() for e in self.partition_by]
        if self.order_by is not None:
            d["order_by"] = [o.to_dict() for o in self.order_by]
        if self.frame is not None:
            d["frame"] = self.frame.to_dict()
        return d


@dataclass
class WindowSpec:
    name: str | None = None
    base: str | None = None
    partition_by: list | None = None
    order_by: list[OrderByItem] | None = None
    frame: FrameSpec | None = None
    filter: Any | None = None

    def to_dict(self) -> dict:
        d: dict = {"name": self.name, "base": self.base}
        if self.partition_by is not None:
            d["partition_by"] = [e.to_dict() for e in self.partition_by]
        if self.order_by is not None:
            d["order_by"] = [o.to_dict() for o in self.order_by]
        if self.frame is not None:
            d["frame"] = self.frame.to_dict()
        if self.filter is not None:
            d["filter"] = self.filter.to_dict()
        return d

    def tables_referenced(self) -> list[str]:
        tables: list[str] = []
        if self.filter is not None:
            tables.extend(self.filter.tables_referenced())
        if self.partition_by:
            for e in self.partition_by:
                tables.extend(e.tables_referenced())
        if self.order_by:
            for item in self.order_by:
                tables.extend(item.expr.tables_referenced())
        return tables

    def functions_used(self) -> list[str]:
        funcs: list[str] = []
        if self.filter is not None:
            funcs.extend(self.filter.functions_used())
        if self.partition_by:
            for e in self.partition_by:
                funcs.extend(e.functions_used())
        if self.order_by:
            for item in self.order_by:
                funcs.extend(item.expr.functions_used())
        return funcs


@dataclass
class FrameSpec:
    type: str  # "RANGE", "ROWS", "GROUPS"
    start: FrameBound
    end: FrameBound

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "start": self.start.to_dict(),
            "end": self.end.to_dict(),
        }


@dataclass
class FrameBound:
    type: str  # "UNBOUNDED", "CURRENT ROW", "PRECEDING", "FOLLOWING"
    expr: Any | None = None

    def to_dict(self) -> dict:
        d: dict = {"type": self.type}
        if self.expr is not None:
            d["expr"] = self.expr.to_dict()
        return d


@dataclass
class Select:
    distinct: bool = False
    all: bool = False
    with_ctes: list[CTE] | None = None
    columns: list[ResultColumn] = field(default_factory=list)
    from_clause: list | None = None
    where: Any | None = None
    group_by: list | None = None
    having: Any | None = None
    window_definitions: list[WindowDef] | None = None
    order_by: list[OrderByItem] | None = None
    limit: Any | None = None
    offset: Any | None = None
    _has_limit: bool = False  # track whether LIMIT was present at all
    _compound_member: bool = False  # True when inside a compound select body

    def to_dict(self) -> dict:
        d: dict = {"type": "select", "distinct": self.distinct, "all": self.all}
        if self.with_ctes is not None:
            d["with"] = [c.to_dict() for c in self.with_ctes]
        d["columns"] = [c.to_dict() for c in self.columns]
        d["from"] = (
            [f.to_dict() for f in self.from_clause]
            if self.from_clause is not None
            else None
        )
        d["where"] = self.where.to_dict() if self.where else None
        d["group_by"] = (
            [e.to_dict() for e in self.group_by]
            if self.group_by is not None
            else None
        )
        d["having"] = self.having.to_dict() if self.having else None
        if self.window_definitions is not None:
            d["window_definitions"] = [w.to_dict() for w in self.window_definitions]
        if not self._compound_member:
            d["order_by"] = (
                [o.to_dict() for o in self.order_by]
                if self.order_by is not None
                else None
            )
            d["limit"] = self.limit.to_dict() if self.limit else None
            if self._has_limit:
                d["offset"] = self.offset.to_dict() if self.offset else None
        return d

    def tables_referenced(self) -> list[str]:
        """Return all table names referenced anywhere in this SELECT."""
        tables: list[str] = []
        if self.with_ctes:
            for cte in self.with_ctes:
                tables.extend(cte.tables_referenced())
        if self.from_clause:
            for item in self.from_clause:
                tables.extend(item.tables_referenced())
        for col in self.columns:
            tables.extend(col.expr.tables_referenced())
        if self.where:
            tables.extend(self.where.tables_referenced())
        if self.having:
            tables.extend(self.having.tables_referenced())
        return tables

    def functions_used(self) -> list[str]:
        """Return all function names used in this SELECT's column expressions."""
        funcs: list[str] = []
        for col in self.columns:
            funcs.extend(col.expr.functions_used())
        return funcs

    def _augmented_columns_for_table(
        self,
        columns_for_table: Callable[[str], list[str]] | None = None,
    ) -> Callable[[str], list[str]]:
        """Build a callback augmented with CTE and subquery-in-FROM schemas."""
        extra: dict[str, list[str]] = {}

        def augmented(table: str) -> list[str]:
            if table in extra:
                return extra[table]
            if columns_for_table:
                return columns_for_table(table)
            return []

        # Process CTEs in declaration order
        if self.with_ctes:
            for cte in self.with_ctes:
                extra[cte.name] = [
                    c.column for c in cte.output_columns(augmented)
                ]

        # Process subqueries in FROM
        if self.from_clause:
            for item in self.from_clause:
                if isinstance(item, SubqueryRef) and item.alias:
                    extra[item.alias] = [
                        c.column for c in item.output_columns(augmented)
                    ]

        return augmented

    def _alias_map(self) -> dict[str, str]:
        """Build {alias_or_name: table_name} from the FROM clause."""
        mapping: dict[str, str] = {}
        if self.from_clause is None:
            return mapping
        for item in self.from_clause:
            if isinstance(item, TableRef):
                key = item.alias if item.alias else item.name
                mapping[key] = item.name
            elif isinstance(item, SubqueryRef) and item.alias:
                mapping[item.alias] = item.alias
        return mapping

    def output_columns(
        self,
        columns_for_table: Callable[[str], list[str]] | None = None,
    ) -> list[OutputColumn]:
        """Return the columns this SELECT produces.

        Args:
            columns_for_table: Optional callback returning column names for a
                given table name. Used to expand ``SELECT *`` and
                ``SELECT t.*``. CTE and subquery-in-FROM schemas are
                inferred automatically and augment this callback.
        """
        augmented = self._augmented_columns_for_table(columns_for_table)
        alias_map = self._alias_map()

        result: list[OutputColumn] = []
        for col in self.columns:
            if col.alias:
                result.append(OutputColumn(table=None, column=col.alias))
            elif isinstance(col.expr, Name):
                result.append(OutputColumn(table=None, column=col.expr.name))
            elif isinstance(col.expr, Dot):
                if isinstance(col.expr.right, Name):
                    table = alias_map.get(
                        col.expr.left.name, col.expr.left.name,
                    ) if isinstance(col.expr.left, Name) else None
                    result.append(OutputColumn(
                        table=table, column=col.expr.right.name,
                    ))
                elif isinstance(col.expr.right, Star):
                    if isinstance(col.expr.left, Name):
                        table = alias_map.get(
                            col.expr.left.name, col.expr.left.name,
                        )
                        cols = augmented(table)
                        if cols:
                            result.extend(
                                OutputColumn(table=table, column=c)
                                for c in cols
                            )
                        else:
                            result.append(OutputColumn(table=table, column="*"))
                else:
                    result.append(OutputColumn(
                        table=None, column=_expr_column_name(col.expr),
                    ))
            elif isinstance(col.expr, Star):
                has_any = False
                for real_table in dict.fromkeys(alias_map.values()):
                    cols = augmented(real_table)
                    if cols:
                        result.extend(
                            OutputColumn(table=real_table, column=c)
                            for c in cols
                        )
                        has_any = True
                if not has_any:
                    result.append(OutputColumn(table=None, column="*"))
            else:
                result.append(OutputColumn(
                    table=None, column=_expr_column_name(col.expr),
                ))
        return result


@dataclass
class Compound:
    body: list  # list of CompoundPart
    order_by: list[OrderByItem] | None = None
    limit: Any | None = None
    offset: Any | None = None
    _has_limit: bool = False

    def to_dict(self) -> dict:
        d: dict = {"type": "compound"}
        d["body"] = [part.to_dict() for part in self.body]
        d["order_by"] = (
            [o.to_dict() for o in self.order_by]
            if self.order_by is not None
            else None
        )
        d["limit"] = self.limit.to_dict() if self.limit else None
        if self._has_limit:
            d["offset"] = self.offset.to_dict() if self.offset else None
        return d

    def tables_referenced(self) -> list[str]:
        tables: list[str] = []
        for part in self.body:
            tables.extend(part.tables_referenced())
        return tables

    def functions_used(self) -> list[str]:
        funcs: list[str] = []
        for part in self.body:
            funcs.extend(part.select.functions_used())
        return funcs

    def output_columns(
        self,
        columns_for_table: Callable[[str], list[str]] | None = None,
    ) -> list[OutputColumn]:
        """Column names come from the first SELECT (SQL standard)."""
        return self.body[0].select.output_columns(columns_for_table)


@dataclass
class CompoundPart:
    select: Select
    operator: str | None = None  # None for first, "UNION"/"UNION ALL"/etc. for rest

    def to_dict(self) -> dict:
        d: dict = {}
        if self.operator is not None:
            d["operator"] = self.operator
        d["select"] = self.select.to_dict()
        return d

    def tables_referenced(self) -> list[str]:
        return self.select.tables_referenced()
