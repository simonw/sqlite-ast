"""SQLite SQL tokenizer."""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto


class TokenType(Enum):
    # Literals
    INTEGER = auto()
    FLOAT = auto()
    STRING = auto()
    BLOB = auto()

    # Identifier
    ID = auto()

    # Parameter
    PARAMETER = auto()

    # Keywords
    SELECT = auto()
    FROM = auto()
    WHERE = auto()
    GROUP = auto()
    BY = auto()
    HAVING = auto()
    ORDER = auto()
    LIMIT = auto()
    OFFSET = auto()
    AS = auto()
    ON = auto()
    USING = auto()
    JOIN = auto()
    LEFT = auto()
    RIGHT = auto()
    FULL = auto()
    OUTER = auto()
    INNER = auto()
    CROSS = auto()
    NATURAL = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    IS = auto()
    IN = auto()
    BETWEEN = auto()
    LIKE = auto()
    GLOB = auto()
    MATCH = auto()
    CASE = auto()
    WHEN = auto()
    THEN = auto()
    ELSE = auto()
    END = auto()
    CAST = auto()
    EXISTS = auto()
    DISTINCT = auto()
    ALL = auto()
    UNION = auto()
    INTERSECT = auto()
    EXCEPT = auto()
    WITH = auto()
    RECURSIVE = auto()
    MATERIALIZED = auto()
    NULL = auto()
    ESCAPE = auto()
    COLLATE = auto()
    ISNULL = auto()
    NOTNULL = auto()
    WINDOW = auto()
    OVER = auto()
    PARTITION = auto()
    RANGE = auto()
    ROWS = auto()
    GROUPS = auto()
    UNBOUNDED = auto()
    PRECEDING = auto()
    FOLLOWING = auto()
    CURRENT = auto()
    ROW = auto()
    EXCLUDE = auto()
    OTHERS = auto()
    TIES = auto()
    FILTER = auto()
    VALUES = auto()
    NULLS = auto()
    FIRST = auto()
    LAST = auto()

    # Operators
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    REM = auto()
    EQ = auto()
    NE = auto()
    LT = auto()
    LE = auto()
    GT = auto()
    GE = auto()
    BITAND = auto()
    BITOR = auto()
    BITNOT = auto()
    LSHIFT = auto()
    RSHIFT = auto()
    CONCAT = auto()

    # Punctuation
    LPAREN = auto()
    RPAREN = auto()
    COMMA = auto()
    DOT = auto()
    SEMICOLON = auto()

    # Special
    EOF = auto()


# Keywords that can fall back to identifiers (from SQLite's %fallback ID)
_FALLBACK_KEYWORDS = {
    "ABORT", "ACTION", "AFTER", "ANALYZE", "ASC", "ATTACH", "BEFORE", "BEGIN",
    "BY", "CASCADE", "CAST", "CONFLICT", "DATABASE", "DEFERRED", "DESC",
    "DETACH", "DO", "EACH", "END", "EXCLUSIVE", "EXPLAIN", "FAIL", "FOR",
    "IGNORE", "IMMEDIATE", "INITIALLY", "INSTEAD", "LIKE", "GLOB", "MATCH",
    "NO", "PLAN", "QUERY", "KEY", "OF", "OFFSET", "PRAGMA", "RAISE",
    "RECURSIVE", "RELEASE", "REPLACE", "RESTRICT", "ROW", "ROWS", "ROLLBACK",
    "SAVEPOINT", "TEMP", "TRIGGER", "VACUUM", "VIEW", "VIRTUAL", "WITH",
    "WITHOUT", "NULLS", "FIRST", "LAST", "CURRENT", "FOLLOWING", "PARTITION",
    "PRECEDING", "RANGE", "UNBOUNDED", "EXCLUDE", "GROUPS", "OTHERS", "TIES",
    "MATERIALIZED", "TRUE", "FALSE",
}

# Map keyword text -> TokenType for keywords we recognize
_KEYWORD_MAP: dict[str, TokenType] = {}
for _tt in TokenType:
    if _tt.name in (
        "SELECT", "FROM", "WHERE", "GROUP", "BY", "HAVING", "ORDER", "LIMIT",
        "OFFSET", "AS", "ON", "USING", "JOIN", "LEFT", "RIGHT", "FULL",
        "OUTER", "INNER", "CROSS", "NATURAL", "AND", "OR", "NOT", "IS", "IN",
        "BETWEEN", "LIKE", "GLOB", "MATCH", "CASE", "WHEN", "THEN", "ELSE",
        "END", "CAST", "EXISTS", "DISTINCT", "ALL", "UNION", "INTERSECT",
        "EXCEPT", "WITH", "RECURSIVE", "MATERIALIZED", "NULL", "ESCAPE",
        "COLLATE", "ISNULL", "NOTNULL", "WINDOW", "OVER", "PARTITION",
        "RANGE", "ROWS", "GROUPS", "UNBOUNDED", "PRECEDING", "FOLLOWING",
        "CURRENT", "ROW", "EXCLUDE", "OTHERS", "TIES", "FILTER", "VALUES",
        "NULLS", "FIRST", "LAST",
    ):
        _KEYWORD_MAP[_tt.name] = _tt


@dataclass(slots=True)
class Token:
    type: TokenType
    value: str  # original text
    pos: int  # byte offset in source


def tokenize(sql: str) -> list[Token]:
    """Tokenize a SQL string into a list of tokens."""
    tokens: list[Token] = []
    i = 0
    n = len(sql)

    while i < n:
        c = sql[i]

        # Skip whitespace
        if c in (" ", "\t", "\n", "\r", "\f"):
            i += 1
            continue

        # Line comments
        if c == "-" and i + 1 < n and sql[i + 1] == "-":
            i += 2
            while i < n and sql[i] != "\n":
                i += 1
            continue

        # Block comments
        if c == "/" and i + 1 < n and sql[i + 1] == "*":
            i += 2
            while i + 1 < n and not (sql[i] == "*" and sql[i + 1] == "/"):
                i += 1
            i += 2  # skip */
            continue

        # String literals
        if c == "'":
            start = i
            i += 1
            parts = []
            while i < n:
                if sql[i] == "'":
                    if i + 1 < n and sql[i + 1] == "'":
                        parts.append(sql[start + 1 : i])
                        parts.append("'")
                        i += 2
                        start = i - 1  # adjust so next slice starts right
                    else:
                        parts.append(sql[start + 1 : i])
                        i += 1
                        break
                else:
                    i += 1
            value = "".join(parts)
            tokens.append(Token(TokenType.STRING, value, start))
            continue

        # Blob literals X'...'
        if c in ("X", "x") and i + 1 < n and sql[i + 1] == "'":
            start = i
            i += 2
            while i < n and sql[i] != "'":
                i += 1
            i += 1  # skip closing quote
            tokens.append(Token(TokenType.BLOB, sql[start:i], start))
            continue

        # Numbers
        if c.isdigit() or (c == "." and i + 1 < n and sql[i + 1].isdigit()):
            start = i
            is_float = False

            # Hex literal
            if c == "0" and i + 1 < n and sql[i + 1] in ("x", "X"):
                i += 2
                while i < n and (sql[i].isdigit() or sql[i] in "abcdefABCDEF"):
                    i += 1
                tokens.append(Token(TokenType.INTEGER, sql[start:i], start))
                continue

            # Integer/float
            while i < n and sql[i].isdigit():
                i += 1
            if i < n and sql[i] == ".":
                is_float = True
                i += 1
                while i < n and sql[i].isdigit():
                    i += 1
            if i < n and sql[i] in ("e", "E"):
                is_float = True
                i += 1
                if i < n and sql[i] in ("+", "-"):
                    i += 1
                while i < n and sql[i].isdigit():
                    i += 1

            text = sql[start:i]
            if is_float:
                tokens.append(Token(TokenType.FLOAT, text, start))
            else:
                tokens.append(Token(TokenType.INTEGER, text, start))
            continue

        # Parameters
        if c == "?":
            start = i
            i += 1
            while i < n and sql[i].isdigit():
                i += 1
            tokens.append(Token(TokenType.PARAMETER, sql[start:i], start))
            continue

        if c in (":", "@", "$"):
            start = i
            i += 1
            while i < n and (sql[i].isalnum() or sql[i] == "_"):
                i += 1
            tokens.append(Token(TokenType.PARAMETER, sql[start:i], start))
            continue

        # Identifiers and keywords
        if c.isalpha() or c == "_":
            start = i
            i += 1
            while i < n and (sql[i].isalnum() or sql[i] == "_"):
                i += 1
            text = sql[start:i]
            upper = text.upper()

            # Check for blob literal (X followed by quote was handled above)
            # Check for keywords
            if upper in _KEYWORD_MAP:
                tokens.append(Token(_KEYWORD_MAP[upper], text, start))
            else:
                tokens.append(Token(TokenType.ID, text, start))
            continue

        # Quoted identifiers "..."
        if c == '"':
            start = i
            i += 1
            while i < n and sql[i] != '"':
                i += 1
            value = sql[start + 1 : i]
            i += 1  # skip closing quote
            tokens.append(Token(TokenType.ID, value, start))
            continue

        # Bracketed identifiers [...]
        if c == "[":
            start = i
            i += 1
            while i < n and sql[i] != "]":
                i += 1
            value = sql[start + 1 : i]
            i += 1  # skip closing bracket
            tokens.append(Token(TokenType.ID, value, start))
            continue

        # Backtick identifiers `...`
        if c == "`":
            start = i
            i += 1
            while i < n and sql[i] != "`":
                i += 1
            value = sql[start + 1 : i]
            i += 1
            tokens.append(Token(TokenType.ID, value, start))
            continue

        # Two-character operators
        if i + 1 < n:
            two = sql[i : i + 2]
            if two == "<=":
                tokens.append(Token(TokenType.LE, two, i))
                i += 2
                continue
            if two == ">=":
                tokens.append(Token(TokenType.GE, two, i))
                i += 2
                continue
            if two == "!=":
                tokens.append(Token(TokenType.NE, two, i))
                i += 2
                continue
            if two == "<>":
                tokens.append(Token(TokenType.NE, two, i))
                i += 2
                continue
            if two == "<<":
                tokens.append(Token(TokenType.LSHIFT, two, i))
                i += 2
                continue
            if two == ">>":
                tokens.append(Token(TokenType.RSHIFT, two, i))
                i += 2
                continue
            if two == "||":
                tokens.append(Token(TokenType.CONCAT, two, i))
                i += 2
                continue
            if two == "==":
                tokens.append(Token(TokenType.EQ, two, i))
                i += 2
                continue

        # Single-character operators and punctuation
        _single = {
            "+": TokenType.PLUS,
            "-": TokenType.MINUS,
            "*": TokenType.STAR,
            "/": TokenType.SLASH,
            "%": TokenType.REM,
            "=": TokenType.EQ,
            "<": TokenType.LT,
            ">": TokenType.GT,
            "&": TokenType.BITAND,
            "|": TokenType.BITOR,
            "~": TokenType.BITNOT,
            "(": TokenType.LPAREN,
            ")": TokenType.RPAREN,
            ",": TokenType.COMMA,
            ".": TokenType.DOT,
            ";": TokenType.SEMICOLON,
        }
        if c in _single:
            tokens.append(Token(_single[c], c, i))
            i += 1
            continue

        # Unknown character — skip
        i += 1

    tokens.append(Token(TokenType.EOF, "", n))
    return tokens
