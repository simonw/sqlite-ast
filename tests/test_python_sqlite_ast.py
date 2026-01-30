import json
import pytest
from pathlib import Path

from sqlite_ast_conformance import AST_TESTS_DIR
from sqlite_ast import parse, ParseError


def load_conformance_tests():
    """Load all conformance test fixtures."""
    tests = []
    for path in sorted(Path(AST_TESTS_DIR).glob("*.json")):
        data = json.loads(path.read_text())
        tests.append(pytest.param(data["sql"], data["ast"], id=path.stem))
    return tests


@pytest.mark.parametrize("sql,expected_ast", load_conformance_tests())
def test_conformance(sql, expected_ast):
    result = parse(sql)
    assert result == expected_ast
