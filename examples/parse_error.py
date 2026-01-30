from pprint import pprint
from sqlite_ast import parse, ParseError

try:
    parse("select 1 union select")
except ParseError as e:
    print(e)
    print("\nPartial AST:")
    pprint(e.partial_ast)
