"""AST node dataclasses for SQLite SELECT statements."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


# --- Expression nodes ---

@dataclass
class IntegerLiteral:
    value: int

    def to_dict(self) -> dict:
        return {"type": "integer", "value": self.value}


@dataclass
class FloatLiteral:
    value: str  # original text representation

    def to_dict(self) -> dict:
        return {"type": "float", "value": self.value}


@dataclass
class StringLiteral:
    value: str

    def to_dict(self) -> dict:
        return {"type": "string", "value": self.value}


@dataclass
class BlobLiteral:
    value: str  # full X'...' text

    def to_dict(self) -> dict:
        return {"type": "blob", "value": self.value}


@dataclass
class NullLiteral:
    def to_dict(self) -> dict:
        return {"type": "null"}


@dataclass
class Name:
    name: str

    def to_dict(self) -> dict:
        return {"type": "name", "name": self.name}


@dataclass
class Star:
    def to_dict(self) -> dict:
        return {"type": "star"}


@dataclass
class Parameter:
    name: str

    def to_dict(self) -> dict:
        return {"type": "parameter", "name": self.name}


@dataclass
class UnaryOp:
    op: str
    operand: Any  # expression node

    def to_dict(self) -> dict:
        return {"type": "unary", "op": self.op, "operand": self.operand.to_dict()}


@dataclass
class BinaryOp:
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


@dataclass
class Dot:
    left: Any
    right: Any

    def to_dict(self) -> dict:
        return {
            "type": "dot",
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
        }


@dataclass
class FunctionCall:
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


@dataclass
class Cast:
    expr: Any
    as_type: str

    def to_dict(self) -> dict:
        return {"type": "cast", "expr": self.expr.to_dict(), "as": self.as_type}


@dataclass
class Case:
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


@dataclass
class Between:
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


@dataclass
class InList:
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


@dataclass
class Exists:
    select: Any

    def to_dict(self) -> dict:
        return {"type": "exists", "select": self.select.to_dict()}


@dataclass
class Subquery:
    select: Any

    def to_dict(self) -> dict:
        return {"type": "subquery", "select": self.select.to_dict()}


@dataclass
class Collate:
    expr: Any
    collation: str

    def to_dict(self) -> dict:
        return {
            "type": "collate",
            "expr": self.expr.to_dict(),
            "collation": self.collation,
        }


@dataclass
class IsNull:
    operand: Any

    def to_dict(self) -> dict:
        return {"type": "isnull", "operand": self.operand.to_dict()}


@dataclass
class NotNull:
    operand: Any

    def to_dict(self) -> dict:
        return {"type": "notnull", "operand": self.operand.to_dict()}


@dataclass
class Unknown:
    """Placeholder node (used for JOIN USING internal representation)."""
    op: int

    def to_dict(self) -> dict:
        return {"type": "unknown", "op": self.op}


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
