import json
from sqlite_ast import parse

ast = parse("select 1")
print(json.dumps(ast, indent=2))
