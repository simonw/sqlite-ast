from pprint import pprint
from sqlite_ast import parse_ast

node = parse_ast("select 1")
pprint(node)
