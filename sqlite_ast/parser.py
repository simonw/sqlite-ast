"""Recursive descent parser for SQLite SELECT statements."""

from __future__ import annotations

from .tokenizer import Token, TokenType, tokenize
from . import ast_nodes as ast


class ParseError(Exception):
    """Raised when SQL parsing fails."""

    def __init__(self, message, partial_ast=None):
        super().__init__(message)
        self._partial_ast = partial_ast

    @property
    def partial_ast(self):
        return self._partial_ast


def parse(sql: str) -> dict:
    """Parse a SQLite SELECT statement and return the AST as a dict."""
    node = parse_ast(sql)
    return node.to_dict()


def parse_ast(sql: str) -> ast.Select | ast.Compound:
    """Parse a SQLite SELECT statement and return the AST as dataclass nodes."""
    tokens = tokenize(sql)
    parser = Parser(tokens, sql)
    return parser.parse_statement()


# --- Operator precedence levels (lowest to highest) ---
# From SQLite parse.y:
#   %left OR.
#   %left AND.
#   %right NOT.
#   %left IS MATCH LIKE_KW BETWEEN IN ISNULL NOTNULL NE EQ.
#   %left GT LE LT GE.
#   %right ESCAPE.
#   %left BITAND BITOR LSHIFT RSHIFT.
#   %left PLUS MINUS.
#   %left STAR SLASH REM.
#   %left CONCAT PTR.
#   %left COLLATE.
#   %right BITNOT.

PREC_OR = 1
PREC_AND = 2
PREC_NOT = 3
PREC_COMPARE = 4  # IS, LIKE, BETWEEN, IN, ISNULL, NOTNULL, NE, EQ
PREC_INEQUALITY = 5  # GT, LE, LT, GE
PREC_BITWISE = 6  # &, |, <<, >>
PREC_ADD = 7  # +, -
PREC_MUL = 8  # *, /, %
PREC_CONCAT = 9  # ||
PREC_COLLATE = 10
PREC_UNARY = 11  # ~, unary -, unary +


class Parser:
    def __init__(self, tokens: list[Token], sql: str):
        self.tokens = tokens
        self.sql = sql
        self.pos = 0
        # Best-effort partial AST: last successfully parsed SELECT node.
        self._last_good_ast: ast.Select | ast.Compound | None = None

    # --- Token helpers ---

    def peek(self) -> Token:
        return self.tokens[self.pos]

    def peek_type(self) -> TokenType:
        return self.tokens[self.pos].type

    def advance(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def expect(self, tt: TokenType) -> Token:
        tok = self.peek()
        if tok.type != tt:
            self._error(f"Expected {tt.name}, got {tok.type.name} ({tok.value!r})")
        return self.advance()

    def match(self, *types: TokenType) -> Token | None:
        if self.peek_type() in types:
            return self.advance()
        return None

    def at(self, *types: TokenType) -> bool:
        return self.peek_type() in types

    def _keyword_as_id(self) -> Token | None:
        """Consume the current token as an identifier if it's a fallback keyword."""
        tok = self.peek()
        if tok.type != TokenType.ID and tok.value.upper() in _ID_FALLBACK_SET:
            return self.advance()
        return None

    def _expect_name(self) -> str:
        """Expect an identifier or keyword-as-identifier. Return its text."""
        tok = self.match(TokenType.ID)
        if tok:
            return tok.value
        tok = self._keyword_as_id()
        if tok:
            return tok.value
        self._error(
            f"Expected identifier, got {self.peek().type.name} ({self.peek().value!r})"
        )

    def _error(self, msg: str):
        tok = self.peek()
        raise ParseError(
            f"Parse error at position {tok.pos}: {msg}",
            partial_ast=self._last_good_ast,
        )

    # --- Top-level ---

    def parse_statement(self) -> ast.Select | ast.Compound:
        result = self._parse_select_or_compound_with_cte()
        if self.peek_type() == TokenType.SEMICOLON:
            self.advance()
        return result

    def _parse_select_or_compound_with_cte(self) -> ast.Select | ast.Compound:
        ctes = None
        if self.at(TokenType.WITH):
            ctes = self._parse_with()

        if self.at(TokenType.VALUES):
            return self._parse_values()

        result = self._parse_select_or_compound()

        # Attach CTEs to the result
        if ctes is not None:
            if isinstance(result, ast.Select):
                result.with_ctes = ctes
            else:
                # Compound: attach CTEs to first select
                result.body[0].select.with_ctes = ctes
        return result

    def _parse_select_or_compound(self) -> ast.Select | ast.Compound:
        first = self._parse_simple_select(in_compound=False)

        # Check for compound operators
        if not self.at(
            TokenType.UNION, TokenType.INTERSECT, TokenType.EXCEPT
        ):
            return first

        first._compound_member = True
        parts = [ast.CompoundPart(select=first, operator=None)]
        while self.at(TokenType.UNION, TokenType.INTERSECT, TokenType.EXCEPT):
            op_tok = self.advance()
            operator = op_tok.value.upper()
            if operator == "UNION" and self.match(TokenType.ALL):
                operator = "UNION ALL"
            next_select = self._parse_simple_select(in_compound=True)
            next_select._compound_member = True
            parts.append(ast.CompoundPart(select=next_select, operator=operator))

        compound = ast.Compound(body=parts)

        # ORDER BY, LIMIT on the compound
        if self.match(TokenType.ORDER):
            self.expect(TokenType.BY)
            compound.order_by = self._parse_order_by_list()

        if self.match(TokenType.LIMIT):
            compound._has_limit = True
            compound.limit = self._parse_expr()
            if self.match(TokenType.OFFSET):
                compound.offset = self._parse_expr()

        return compound

    # --- Simple SELECT ---

    def _parse_simple_select(self, in_compound: bool = False) -> ast.Select:
        self.expect(TokenType.SELECT)
        sel = ast.Select()

        # DISTINCT / ALL
        if self.match(TokenType.DISTINCT):
            sel.distinct = True
        elif self.match(TokenType.ALL):
            sel.all = True

        # Columns
        sel.columns = self._parse_result_columns()

        # FROM
        if self.match(TokenType.FROM):
            sel.from_clause = self._parse_from_clause()

        # WHERE
        if self.match(TokenType.WHERE):
            sel.where = self._parse_expr()

        # GROUP BY
        if self.peek_type() == TokenType.GROUP:
            self.advance()
            self.expect(TokenType.BY)
            sel.group_by = self._parse_expr_list()

        # HAVING
        if self.match(TokenType.HAVING):
            sel.having = self._parse_expr()

        # WINDOW clause
        if self.match(TokenType.WINDOW):
            sel.window_definitions = self._parse_window_definitions()

        # ORDER BY and LIMIT are only parsed for standalone selects,
        # not for compound members (compound handles its own ORDER BY/LIMIT)
        if not in_compound:
            if self.peek().value.upper() == "ORDER":
                self.advance()
                self.expect(TokenType.BY)
                sel.order_by = self._parse_order_by_list()

            if self.peek().value.upper() == "LIMIT":
                self.advance()
                sel._has_limit = True
                sel.limit = self._parse_expr()
                if self.match(TokenType.OFFSET):
                    sel.offset = self._parse_expr()

        # Record the last fully parsed SELECT as a partial AST candidate.
        self._last_good_ast = sel
        return sel

    # --- Result columns ---

    def _parse_result_columns(self) -> list[ast.ResultColumn]:
        cols = [self._parse_result_column()]
        while self.match(TokenType.COMMA):
            cols.append(self._parse_result_column())
        return cols

    def _parse_result_column(self) -> ast.ResultColumn:
        expr = self._parse_expr()
        alias = None
        if self.match(TokenType.AS):
            alias = self._expect_name()
        elif self.at(TokenType.ID) or (
            self.peek().value.upper() in _ID_FALLBACK_SET
            and self.peek_type() != TokenType.FROM
            and self.peek_type() != TokenType.WHERE
            and self.peek_type() != TokenType.GROUP
            and self.peek_type() != TokenType.HAVING
            and self.peek_type() != TokenType.ORDER
            and self.peek_type() != TokenType.LIMIT
            and self.peek_type() != TokenType.WINDOW
            and self.peek_type() != TokenType.UNION
            and self.peek_type() != TokenType.INTERSECT
            and self.peek_type() != TokenType.EXCEPT
        ):
            # Implicit alias (no AS keyword)
            alias = self._expect_name()
        return ast.ResultColumn(expr=expr, alias=alias)

    # --- FROM clause ---

    def _parse_from_clause(self) -> list:
        items = [self._parse_from_item(is_first=True)]

        while True:
            join_type = self._try_parse_join_type()
            if join_type is None:
                # Check for comma-separated tables (implicit JOIN)
                if self.match(TokenType.COMMA):
                    join_type = "JOIN"
                else:
                    break

            item = self._parse_from_item(is_first=False)
            self._set_join_type(item, join_type)

            # ON / USING
            if self.match(TokenType.ON):
                on_expr = self._parse_expr()
                self._set_on(item, on_expr)
            if self.match(TokenType.USING):
                self.expect(TokenType.LPAREN)
                cols = [self._expect_name()]
                while self.match(TokenType.COMMA):
                    cols.append(self._expect_name())
                self.expect(TokenType.RPAREN)
                # USING produces both an "on" placeholder and "using" list
                self._set_on(item, ast.Unknown(op=1))
                self._set_using(item, cols)

            items.append(item)

        return items

    def _parse_from_item(self, is_first: bool) -> ast.TableRef | ast.SubqueryRef:
        if self.match(TokenType.LPAREN):
            # Subquery or parenthesized table expression
            if self.at(TokenType.SELECT, TokenType.WITH, TokenType.VALUES):
                sub = self._parse_select_or_compound_with_cte()
                self.expect(TokenType.RPAREN)
                alias = None
                if self.match(TokenType.AS):
                    alias = self._expect_name()
                elif self.at(TokenType.ID) or self._is_id_fallback():
                    alias = self._expect_name()
                return ast.SubqueryRef(select=sub, alias=alias)
            else:
                # Could be parenthesized join — for now treat as subquery
                sub = self._parse_select_or_compound_with_cte()
                self.expect(TokenType.RPAREN)
                alias = None
                if self.match(TokenType.AS):
                    alias = self._expect_name()
                return ast.SubqueryRef(select=sub, alias=alias)

        name = self._expect_name()
        alias = None
        if self.match(TokenType.AS):
            alias = self._expect_name()
        elif self._can_be_implicit_alias():
            alias = self._expect_name()
        return ast.TableRef(name=name, alias=alias)

    def _try_parse_join_type(self) -> str | None:
        """Try to parse a join keyword sequence. Return the join type string or None."""
        tok = self.peek()

        # NATURAL [LEFT|RIGHT|FULL [OUTER]|INNER|CROSS] JOIN
        if tok.type == TokenType.NATURAL:
            self.advance()
            parts = ["NATURAL"]
            if self.match(TokenType.LEFT):
                parts.append("LEFT")
            elif self.match(TokenType.RIGHT):
                parts.append("RIGHT")
            elif self.match(TokenType.FULL):
                parts.append("FULL")
                if self.match(TokenType.OUTER):
                    parts.append("OUTER")
            elif self.match(TokenType.INNER):
                parts.append("INNER")
            elif self.match(TokenType.CROSS):
                parts.append("CROSS")
            self.expect(TokenType.JOIN)
            parts.append("JOIN")
            return " ".join(parts)

        # LEFT [OUTER] JOIN
        if tok.type == TokenType.LEFT:
            self.advance()
            if self.match(TokenType.OUTER):
                self.expect(TokenType.JOIN)
                return "LEFT OUTER JOIN"
            self.expect(TokenType.JOIN)
            return "LEFT JOIN"

        # RIGHT [OUTER] JOIN
        if tok.type == TokenType.RIGHT:
            self.advance()
            if self.match(TokenType.OUTER):
                self.expect(TokenType.JOIN)
                return "RIGHT OUTER JOIN"
            self.expect(TokenType.JOIN)
            return "RIGHT JOIN"

        # FULL [OUTER] JOIN
        if tok.type == TokenType.FULL:
            self.advance()
            if self.match(TokenType.OUTER):
                pass
            self.expect(TokenType.JOIN)
            return "FULL OUTER JOIN"

        # CROSS JOIN
        if tok.type == TokenType.CROSS:
            self.advance()
            self.expect(TokenType.JOIN)
            return "CROSS JOIN"

        # INNER JOIN
        if tok.type == TokenType.INNER:
            self.advance()
            self.expect(TokenType.JOIN)
            return "INNER JOIN"

        # Plain JOIN
        if tok.type == TokenType.JOIN:
            self.advance()
            return "JOIN"

        return None

    def _set_join_type(self, item, jt):
        if isinstance(item, ast.TableRef):
            item.join_type = jt
        elif isinstance(item, ast.SubqueryRef):
            item.join_type = jt

    def _set_on(self, item, expr):
        if isinstance(item, (ast.TableRef, ast.SubqueryRef)):
            item.on = expr

    def _set_using(self, item, cols):
        if isinstance(item, ast.TableRef):
            item.using = cols

    def _is_id_fallback(self) -> bool:
        return self.peek().value.upper() in _ID_FALLBACK_SET and self.peek_type() != TokenType.ID

    def _can_be_implicit_alias(self) -> bool:
        """Check if current token could be an implicit alias (no AS keyword)."""
        if self.at(TokenType.ID):
            return True
        tok = self.peek()
        upper = tok.value.upper()
        # Don't consume structural keywords as implicit aliases
        if upper in (
            "FROM", "WHERE", "GROUP", "HAVING", "ORDER", "LIMIT", "WINDOW",
            "UNION", "INTERSECT", "EXCEPT", "ON", "USING", "JOIN", "LEFT",
            "RIGHT", "FULL", "OUTER", "INNER", "CROSS", "NATURAL",
        ):
            return False
        return upper in _ID_FALLBACK_SET

    # --- ORDER BY ---

    def _parse_order_by_list(self) -> list[ast.OrderByItem]:
        items = [self._parse_order_by_item()]
        while self.match(TokenType.COMMA):
            items.append(self._parse_order_by_item())
        return items

    def _parse_order_by_item(self) -> ast.OrderByItem:
        expr = self._parse_expr()
        direction = "ASC"
        # ASC/DESC are fallback keywords — check by value
        if self.peek().value.upper() == "ASC":
            self.advance()
            direction = "ASC"
        elif self.peek().value.upper() == "DESC":
            self.advance()
            direction = "DESC"

        nulls = None
        if self.match(TokenType.NULLS):
            if self.match(TokenType.FIRST):
                # BIGNULL normalization: FIRST means sort_dir != nulls_val
                # NULLS FIRST = SO_ASC(0). BIGNULL set when dir != nulls.
                # dir=ASC(0) + FIRST(0) → same → no BIGNULL → output "FIRST"
                # dir=DESC(1) + FIRST(0) → diff → BIGNULL → output "LAST"
                nulls_val = 0  # FIRST = SO_ASC = 0
            elif self.match(TokenType.LAST):
                nulls_val = 1  # LAST = SO_DESC = 1
            else:
                self._error("Expected FIRST or LAST after NULLS")
                return ast.OrderByItem(expr=expr, direction=direction)

            dir_val = 0 if direction == "ASC" else 1
            if dir_val != nulls_val:
                nulls = "LAST"   # BIGNULL
            else:
                nulls = "FIRST"  # no BIGNULL

        return ast.OrderByItem(expr=expr, direction=direction, nulls=nulls)

    # --- Expression list ---

    def _parse_expr_list(self) -> list:
        exprs = [self._parse_expr()]
        while self.match(TokenType.COMMA):
            exprs.append(self._parse_expr())
        return exprs

    # --- Expression parsing (Pratt) ---

    def _parse_expr(self, min_prec: int = 0) -> ast.Node:
        left = self._parse_prefix()
        while True:
            prec, right_prec = self._infix_precedence()
            if prec is None or prec < min_prec:
                break
            left = self._parse_infix(left, right_prec)
        return left

    def _parse_prefix(self):
        tok = self.peek()
        tt = tok.type

        # NULL
        if tt == TokenType.NULL:
            self.advance()
            return ast.NullLiteral()

        # Integer
        if tt == TokenType.INTEGER:
            self.advance()
            return ast.IntegerLiteral(value=int(tok.value, 0))

        # Float
        if tt == TokenType.FLOAT:
            self.advance()
            return ast.FloatLiteral(value=tok.value)

        # String
        if tt == TokenType.STRING:
            self.advance()
            return ast.StringLiteral(value=tok.value)

        # Blob
        if tt == TokenType.BLOB:
            self.advance()
            return ast.BlobLiteral(value=tok.value)

        # Parameter
        if tt == TokenType.PARAMETER:
            self.advance()
            return ast.Parameter(name=tok.value)

        # Star
        if tt == TokenType.STAR:
            self.advance()
            return ast.Star()

        # Unary minus
        if tt == TokenType.MINUS:
            self.advance()
            operand = self._parse_expr(PREC_UNARY)
            return ast.UnaryOp(op="-", operand=operand)

        # Unary plus
        if tt == TokenType.PLUS:
            self.advance()
            operand = self._parse_expr(PREC_UNARY)
            return ast.UnaryOp(op="+", operand=operand)

        # Bitwise NOT
        if tt == TokenType.BITNOT:
            self.advance()
            operand = self._parse_expr(PREC_UNARY)
            return ast.UnaryOp(op="~", operand=operand)

        # NOT (unary prefix)
        if tt == TokenType.NOT:
            self.advance()
            # Check for NOT EXISTS
            if self.at(TokenType.EXISTS):
                return ast.UnaryOp(op="NOT", operand=self._parse_prefix())
            operand = self._parse_expr(PREC_NOT)
            return ast.UnaryOp(op="NOT", operand=operand)

        # CAST
        if tt == TokenType.CAST:
            return self._parse_cast()

        # CASE
        if tt == TokenType.CASE:
            return self._parse_case()

        # EXISTS
        if tt == TokenType.EXISTS:
            self.advance()
            self.expect(TokenType.LPAREN)
            sub = self._parse_select_or_compound_with_cte()
            self.expect(TokenType.RPAREN)
            return ast.Exists(select=sub)

        # Parenthesized expression or subquery
        if tt == TokenType.LPAREN:
            self.advance()
            if self.at(TokenType.SELECT, TokenType.WITH, TokenType.VALUES):
                sub = self._parse_select_or_compound_with_cte()
                self.expect(TokenType.RPAREN)
                return ast.Subquery(select=sub)
            expr = self._parse_expr()
            self.expect(TokenType.RPAREN)
            return expr

        # Identifier (or keyword-as-identifier) — might be function call
        if tt == TokenType.ID or tok.value.upper() in _ID_FALLBACK_SET:
            name_tok = self.advance()
            name = name_tok.value

            # Function call?
            if self.at(TokenType.LPAREN):
                return self._parse_function_call(name)

            # Plain identifier
            upper = name.upper()
            if upper in ("TRUE", "FALSE") or name_tok.type == TokenType.ID:
                return ast.Name(name=name)
            else:
                # Keyword used as identifier
                return ast.Name(name=upper if name_tok.type != TokenType.ID else name)

        self._error(
            f"Unexpected token in expression: {tok.type.name} ({tok.value!r})"
        )

    def _parse_function_call(self, name: str) -> ast.FunctionCall | ast.Name:
        self.expect(TokenType.LPAREN)

        # count(*)
        if name.lower() == "count" and self.at(TokenType.STAR):
            self.advance()
            self.expect(TokenType.RPAREN)
            func = ast.FunctionCall(name=name, args=[], distinct=False)
            return self._maybe_parse_over_and_filter(func)

        # No args
        if self.at(TokenType.RPAREN):
            self.advance()
            func = ast.FunctionCall(name=name, args=[], distinct=False)
            return self._maybe_parse_over_and_filter(func)

        # DISTINCT in function args
        distinct = False
        if self.match(TokenType.DISTINCT):
            distinct = True

        args = self._parse_expr_list()
        self.expect(TokenType.RPAREN)
        func = ast.FunctionCall(name=name, args=args, distinct=distinct)
        return self._maybe_parse_over_and_filter(func)

    def _maybe_parse_over_and_filter(self, func: ast.FunctionCall):
        """Parse optional FILTER and OVER clauses on a function call."""
        filter_expr = None
        if self.match(TokenType.FILTER):
            self.expect(TokenType.LPAREN)
            self.expect(TokenType.WHERE)
            filter_expr = self._parse_expr()
            self.expect(TokenType.RPAREN)

        if self.match(TokenType.OVER):
            over = self._parse_window_spec()
            if filter_expr is not None:
                over.filter = filter_expr
            func.over = over

        return func

    def _parse_window_spec(self) -> ast.WindowSpec:
        """Parse a window specification after OVER."""
        # OVER window_name
        if self.at(TokenType.ID) or self._is_id_fallback():
            if not self.at(TokenType.LPAREN):
                name = self._expect_name()
                return ast.WindowSpec(name=name)

        # OVER (...)
        self.expect(TokenType.LPAREN)
        spec = ast.WindowSpec()

        # base window name
        if (self.at(TokenType.ID) or self._is_id_fallback()) and not self.at(
            TokenType.PARTITION, TokenType.ORDER, TokenType.RANGE,
            TokenType.ROWS, TokenType.GROUPS,
        ):
            # Peek ahead to see if this is a base name or a partition/order keyword
            saved = self.pos
            name = self._expect_name()
            if self.at(
                TokenType.PARTITION, TokenType.ORDER, TokenType.RANGE,
                TokenType.ROWS, TokenType.GROUPS, TokenType.RPAREN,
            ):
                spec.base = name
            else:
                self.pos = saved

        # PARTITION BY
        if self.match(TokenType.PARTITION):
            self.expect(TokenType.BY)
            spec.partition_by = self._parse_expr_list()

        # ORDER BY
        if self.peek_type() == TokenType.ORDER:
            self.advance()
            self.expect(TokenType.BY)
            spec.order_by = self._parse_order_by_list()

        # Frame spec
        if self.at(TokenType.RANGE, TokenType.ROWS, TokenType.GROUPS):
            spec.frame = self._parse_frame_spec()
        else:
            # Default frame for any inline window spec
            spec.frame = ast.FrameSpec(
                type="RANGE",
                start=ast.FrameBound(type="UNBOUNDED"),
                end=ast.FrameBound(type="CURRENT ROW"),
            )

        self.expect(TokenType.RPAREN)
        return spec

    def _parse_frame_spec(self) -> ast.FrameSpec:
        frame_type = self.advance().value.upper()

        if self.peek().value.upper() == "BETWEEN" and self.peek_type() == TokenType.BETWEEN:
            self.advance()
            start = self._parse_frame_bound()
            self.expect(TokenType.AND)
            end = self._parse_frame_bound()
        else:
            start = self._parse_frame_bound()
            end = ast.FrameBound(type="CURRENT ROW")

        return ast.FrameSpec(type=frame_type, start=start, end=end)

    def _parse_frame_bound(self) -> ast.FrameBound:
        if self.match(TokenType.UNBOUNDED):
            # UNBOUNDED PRECEDING or UNBOUNDED FOLLOWING
            if self.match(TokenType.PRECEDING):
                return ast.FrameBound(type="UNBOUNDED")
            elif self.match(TokenType.FOLLOWING):
                return ast.FrameBound(type="UNBOUNDED")
            return ast.FrameBound(type="UNBOUNDED")

        if self.match(TokenType.CURRENT):
            self.expect(TokenType.ROW)
            return ast.FrameBound(type="CURRENT ROW")

        # expr PRECEDING or expr FOLLOWING
        expr = self._parse_expr()
        if self.match(TokenType.PRECEDING):
            return ast.FrameBound(type="PRECEDING", expr=expr)
        elif self.match(TokenType.FOLLOWING):
            return ast.FrameBound(type="FOLLOWING", expr=expr)

        self._error("Expected PRECEDING or FOLLOWING")

    # --- Infix precedence ---

    def _infix_precedence(self) -> tuple[int | None, int]:
        """Return (precedence, right_precedence) for the current token as infix op."""
        tok = self.peek()
        tt = tok.type

        if tt == TokenType.OR:
            return (PREC_OR, PREC_OR + 1)  # left-assoc
        if tt == TokenType.AND:
            return (PREC_AND, PREC_AND + 1)
        if tt in (TokenType.EQ, TokenType.NE):
            return (PREC_COMPARE, PREC_COMPARE + 1)
        if tt == TokenType.IS:
            return (PREC_COMPARE, PREC_COMPARE + 1)
        if tt in (TokenType.LIKE, TokenType.GLOB, TokenType.MATCH):
            return (PREC_COMPARE, PREC_COMPARE + 1)
        if tt == TokenType.IN:
            return (PREC_COMPARE, PREC_COMPARE + 1)
        if tt == TokenType.BETWEEN:
            return (PREC_COMPARE, PREC_COMPARE + 1)
        if tt == TokenType.NOT:
            # NOT IN, NOT BETWEEN, NOT LIKE, NOT GLOB, NOT MATCH
            nxt = self.tokens[self.pos + 1] if self.pos + 1 < len(self.tokens) else None
            if nxt and nxt.type in (
                TokenType.IN, TokenType.BETWEEN, TokenType.LIKE,
                TokenType.GLOB, TokenType.MATCH,
            ):
                return (PREC_COMPARE, PREC_COMPARE + 1)
            # NOT NULL (postfix)
            if nxt and nxt.type == TokenType.NULL:
                return (PREC_COMPARE, PREC_COMPARE + 1)
            return (None, 0)
        if tt == TokenType.ISNULL:
            return (PREC_COMPARE, PREC_COMPARE + 1)
        if tt == TokenType.NOTNULL:
            return (PREC_COMPARE, PREC_COMPARE + 1)
        if tt in (TokenType.LT, TokenType.LE, TokenType.GT, TokenType.GE):
            return (PREC_INEQUALITY, PREC_INEQUALITY + 1)
        if tt in (TokenType.BITAND, TokenType.BITOR, TokenType.LSHIFT, TokenType.RSHIFT):
            return (PREC_BITWISE, PREC_BITWISE + 1)
        if tt in (TokenType.PLUS, TokenType.MINUS):
            return (PREC_ADD, PREC_ADD + 1)
        if tt in (TokenType.STAR, TokenType.SLASH, TokenType.REM):
            return (PREC_MUL, PREC_MUL + 1)
        if tt == TokenType.CONCAT:
            return (PREC_CONCAT, PREC_CONCAT + 1)
        if tt == TokenType.COLLATE:
            return (PREC_COLLATE, PREC_COLLATE + 1)
        if tt == TokenType.DOT:
            return (100, 100)  # Highest precedence for dot

        return (None, 0)

    # --- Infix parsing ---

    def _parse_infix(self, left, right_prec: int):
        tok = self.peek()
        tt = tok.type

        # Dot (qualified name)
        if tt == TokenType.DOT:
            self.advance()
            if self.match(TokenType.STAR):
                return ast.Dot(left=left, right=ast.Star())
            right = self._parse_prefix()
            return ast.Dot(left=left, right=right)

        # COLLATE
        if tt == TokenType.COLLATE:
            self.advance()
            name = self._expect_name()
            return ast.Collate(expr=left, collation=name)

        # IS / IS NOT
        if tt == TokenType.IS:
            self.advance()
            negate = False
            if self.match(TokenType.NOT):
                negate = True
                # IS NOT DISTINCT FROM
                if self.match(TokenType.DISTINCT):
                    self.expect(TokenType.FROM)
                    right = self._parse_expr(right_prec)
                    return ast.BinaryOp(op="IS", left=left, right=right)
            else:
                # IS DISTINCT FROM
                if self.match(TokenType.DISTINCT):
                    self.expect(TokenType.FROM)
                    right = self._parse_expr(right_prec)
                    return ast.BinaryOp(op="IS NOT", left=left, right=right)

            right = self._parse_expr(right_prec)

            # IS NULL / IS NOT NULL special handling
            if isinstance(right, ast.NullLiteral):
                if negate:
                    return self._make_notnull(left)
                else:
                    return self._make_isnull(left)

            if negate:
                return ast.BinaryOp(op="IS NOT", left=left, right=right)
            return ast.BinaryOp(op="IS", left=left, right=right)

        # ISNULL (postfix)
        if tt == TokenType.ISNULL:
            self.advance()
            return self._make_isnull(left)

        # NOTNULL (postfix)
        if tt == TokenType.NOTNULL:
            self.advance()
            return self._make_notnull(left)

        # NOT IN / NOT BETWEEN / NOT LIKE / NOT GLOB / NOT MATCH / NOT NULL
        if tt == TokenType.NOT:
            nxt = self.tokens[self.pos + 1]
            if nxt.type == TokenType.NULL:
                self.advance()  # NOT
                self.advance()  # NULL
                return self._make_notnull(left)
            if nxt.type == TokenType.IN:
                self.advance()  # NOT
                self.advance()  # IN
                inner = self._parse_in(left)
                return ast.UnaryOp(op="NOT", operand=inner)
            if nxt.type == TokenType.BETWEEN:
                self.advance()  # NOT
                self.advance()  # BETWEEN
                inner = self._parse_between(left)
                return ast.UnaryOp(op="NOT", operand=inner)
            if nxt.type in (TokenType.LIKE, TokenType.GLOB, TokenType.MATCH):
                self.advance()  # NOT
                like_tok = self.advance()
                inner = self._parse_like(left, like_tok.value.upper())
                return ast.UnaryOp(op="NOT", operand=inner)

        # LIKE / GLOB / MATCH
        if tt in (TokenType.LIKE, TokenType.GLOB, TokenType.MATCH):
            self.advance()
            return self._parse_like(left, tok.value.upper())

        # IN
        if tt == TokenType.IN:
            self.advance()
            return self._parse_in(left)

        # BETWEEN
        if tt == TokenType.BETWEEN:
            self.advance()
            return self._parse_between(left)

        # Binary operators
        op_map = {
            TokenType.OR: "OR",
            TokenType.AND: "AND",
            TokenType.PLUS: "+",
            TokenType.MINUS: "-",
            TokenType.STAR: "*",
            TokenType.SLASH: "/",
            TokenType.REM: "%",
            TokenType.EQ: "=",
            TokenType.NE: "!=",
            TokenType.LT: "<",
            TokenType.LE: "<=",
            TokenType.GT: ">",
            TokenType.GE: ">=",
            TokenType.BITAND: "&",
            TokenType.BITOR: "|",
            TokenType.LSHIFT: "<<",
            TokenType.RSHIFT: ">>",
            TokenType.CONCAT: "||",
        }

        if tt in op_map:
            self.advance()
            op = op_map[tt]
            right = self._parse_expr(right_prec)

            # AND constant folding (SQLite's sqlite3ExprAnd)
            if op == "AND":
                return self._make_and(left, right)

            return ast.BinaryOp(op=op, left=left, right=right)

        self._error(f"Unexpected infix token: {tok.type.name} ({tok.value!r})")

    # --- Special expression forms ---

    def _parse_like(self, left, func_name: str):
        """Parse LIKE/GLOB/MATCH — converts to function call with reversed args."""
        pattern = self._parse_expr(PREC_COMPARE + 1)
        args = [pattern, left]
        if self.match(TokenType.ESCAPE):
            escape = self._parse_expr(PREC_COMPARE + 1)
            args.append(escape)
        return ast.FunctionCall(name=func_name, args=args, distinct=False)

    def _parse_in(self, left):
        """Parse IN (list) or IN (subquery)."""
        self.expect(TokenType.LPAREN)
        if self.at(TokenType.SELECT, TokenType.WITH, TokenType.VALUES):
            sub = self._parse_select_or_compound_with_cte()
            self.expect(TokenType.RPAREN)
            return ast.InList(expr=left, select=sub)
        values = self._parse_expr_list()
        self.expect(TokenType.RPAREN)
        return ast.InList(expr=left, values=values)

    def _parse_between(self, left):
        """Parse BETWEEN low AND high."""
        low = self._parse_expr(PREC_COMPARE + 1)
        self.expect(TokenType.AND)
        high = self._parse_expr(PREC_COMPARE + 1)
        return ast.Between(expr=left, low=low, high=high)

    def _parse_cast(self):
        self.advance()  # CAST
        self.expect(TokenType.LPAREN)
        expr = self._parse_expr()
        self.expect(TokenType.AS)
        # Type name — may be multiple tokens
        type_name = self._expect_name()
        self.expect(TokenType.RPAREN)
        return ast.Cast(expr=expr, as_type=type_name)

    def _parse_case(self):
        self.advance()  # CASE
        operand = None
        # Simple CASE: CASE expr WHEN ...
        # Searched CASE: CASE WHEN ...
        if not self.at(TokenType.WHEN):
            operand = self._parse_expr()

        when_clauses = []
        while self.match(TokenType.WHEN):
            when_expr = self._parse_expr()
            self.expect(TokenType.THEN)
            then_expr = self._parse_expr()
            when_clauses.append((when_expr, then_expr))

        else_expr = None
        if self.match(TokenType.ELSE):
            else_expr = self._parse_expr()

        self.expect(TokenType.END)
        return ast.Case(operand=operand, when_clauses=when_clauses, else_expr=else_expr)

    # --- SQLite-specific constant folding ---

    def _make_and(self, left, right):
        """Create AND node with SQLite's constant folding.

        If either side is integer literal 0, fold to integer 0.
        """
        if isinstance(left, ast.IntegerLiteral) and left.value == 0:
            return ast.IntegerLiteral(value=0)
        if isinstance(right, ast.IntegerLiteral) and right.value == 0:
            return ast.IntegerLiteral(value=0)
        return ast.BinaryOp(op="AND", left=left, right=right)

    def _make_isnull(self, operand):
        """Create ISNULL node with constant folding.

        If operand is a literal (possibly wrapped in unary +/-), fold to integer 0.
        """
        if self._is_literal(operand):
            return ast.IntegerLiteral(value=0)
        return ast.IsNull(operand=operand)

    def _make_notnull(self, operand):
        """Create NOTNULL node with constant folding.

        If operand is a literal (possibly wrapped in unary +/-), fold to integer 1.
        """
        if self._is_literal(operand):
            return ast.IntegerLiteral(value=1)
        return ast.NotNull(operand=operand)

    def _is_literal(self, node) -> bool:
        """Check if node is a literal, looking through unary +/- wrappers."""
        if isinstance(node, (ast.IntegerLiteral, ast.FloatLiteral, ast.StringLiteral, ast.BlobLiteral)):
            return True
        if isinstance(node, ast.UnaryOp) and node.op in ("+", "-"):
            return self._is_literal(node.operand)
        return False

    # --- WITH / CTE ---

    def _parse_with(self) -> list[ast.CTE]:
        self.expect(TokenType.WITH)
        self.match(TokenType.RECURSIVE)  # consume RECURSIVE but don't flag it

        ctes = [self._parse_cte()]
        while self.match(TokenType.COMMA):
            ctes.append(self._parse_cte())
        return ctes

    def _parse_cte(self) -> ast.CTE:
        name = self._expect_name()
        columns = None
        materialized = None

        # Optional column list
        if self.at(TokenType.LPAREN):
            self.advance()
            columns = [self._expect_name()]
            while self.match(TokenType.COMMA):
                columns.append(self._expect_name())
            self.expect(TokenType.RPAREN)

        self.expect(TokenType.AS)

        # MATERIALIZED / NOT MATERIALIZED
        if self.match(TokenType.MATERIALIZED):
            materialized = "MATERIALIZED"
        elif self.at(TokenType.NOT):
            if self.pos + 1 < len(self.tokens) and self.tokens[self.pos + 1].type == TokenType.MATERIALIZED:
                self.advance()  # NOT
                self.advance()  # MATERIALIZED
                materialized = "NOT MATERIALIZED"

        self.expect(TokenType.LPAREN)
        select = self._parse_select_or_compound()
        self.expect(TokenType.RPAREN)

        return ast.CTE(name=name, select=select, columns=columns, materialized=materialized)

    # --- WINDOW clause ---

    def _parse_window_definitions(self) -> list[ast.WindowDef]:
        defs = [self._parse_window_definition()]
        while self.match(TokenType.COMMA):
            defs.append(self._parse_window_definition())
        return defs

    def _parse_window_definition(self) -> ast.WindowDef:
        name = self._expect_name()
        self.expect(TokenType.AS)
        self.expect(TokenType.LPAREN)

        wdef = ast.WindowDef(name=name)

        # base window
        if (self.at(TokenType.ID) or self._is_id_fallback()) and not self.at(
            TokenType.PARTITION, TokenType.ORDER, TokenType.RANGE,
            TokenType.ROWS, TokenType.GROUPS,
        ):
            saved = self.pos
            base = self._expect_name()
            if self.at(
                TokenType.PARTITION, TokenType.ORDER, TokenType.RANGE,
                TokenType.ROWS, TokenType.GROUPS, TokenType.RPAREN,
            ):
                wdef.base = base
            else:
                self.pos = saved

        # PARTITION BY
        if self.match(TokenType.PARTITION):
            self.expect(TokenType.BY)
            wdef.partition_by = self._parse_expr_list()

        # ORDER BY
        if self.peek_type() == TokenType.ORDER:
            self.advance()
            self.expect(TokenType.BY)
            wdef.order_by = self._parse_order_by_list()

        # Frame spec
        if self.at(TokenType.RANGE, TokenType.ROWS, TokenType.GROUPS):
            wdef.frame = self._parse_frame_spec()
        elif wdef.order_by is not None or wdef.partition_by is not None:
            # Default frame when ORDER BY or PARTITION BY present
            wdef.frame = ast.FrameSpec(
                type="RANGE",
                start=ast.FrameBound(type="UNBOUNDED"),
                end=ast.FrameBound(type="CURRENT ROW"),
            )

        self.expect(TokenType.RPAREN)
        return wdef

    # --- VALUES ---

    def _parse_values(self) -> ast.Select:
        """Parse VALUES clause — returns SQLite's internal transformation."""
        self.expect(TokenType.VALUES)
        # Skip all the value rows
        while True:
            self.expect(TokenType.LPAREN)
            self._parse_expr_list()
            self.expect(TokenType.RPAREN)
            if not self.match(TokenType.COMMA):
                break

        return ast.Select(
            columns=[ast.ResultColumn(expr=ast.Star(), alias=None)],
            from_clause=[
                ast.TableRef(name="sqlite_master", schema="main", alias=None)
            ],
            order_by=[
                ast.OrderByItem(expr=ast.Name(name="rowid"), direction="ASC")
            ],
        )


# Set of uppercase keyword strings that can fall back to identifiers
_ID_FALLBACK_SET = {
    "ABORT", "ACTION", "AFTER", "ANALYZE", "ASC", "ATTACH", "BEFORE", "BEGIN",
    "BY", "CASCADE", "CAST", "CONFLICT", "DATABASE", "DEFERRED", "DESC",
    "DETACH", "DO", "EACH", "END", "EXCLUSIVE", "EXPLAIN", "FAIL", "FOR",
    "IGNORE", "IMMEDIATE", "INITIALLY", "INSTEAD", "LIKE", "GLOB", "MATCH",
    "NO", "PLAN", "QUERY", "KEY", "OF", "OFFSET", "PRAGMA", "RAISE",
    "RECURSIVE", "RELEASE", "REPLACE", "RESTRICT", "ROW", "ROWS", "ROLLBACK",
    "SAVEPOINT", "TEMP", "TRIGGER", "VACUUM", "VIEW", "VIRTUAL", "WITH",
    "WITHOUT", "NULLS", "FIRST", "LAST", "CURRENT", "FOLLOWING", "PARTITION",
    "PRECEDING", "RANGE", "UNBOUNDED", "EXCLUDE", "GROUPS", "OTHERS", "TIES",
    "MATERIALIZED", "TRUE", "FALSE", "FILTER", "WINDOW", "OVER",
}
