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
    raise ParseError(f"Not yet implemented", partial_ast=None)
