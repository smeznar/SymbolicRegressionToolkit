"""Print the specification of a module: signatures and docstrings, without function bodies.

Used by the ``spec-tester`` agent (``.claude/agents/spec-tester.md``) so that black-box tests are
designed from the documented contract rather than from the current implementation.

Usage:
    python scripts/spec_view.py SRToolkit/utils/measures.py [more files ...] [--private]

Module-level and class-level assignments are shown with their attribute docstring (a string literal
directly after the assignment); long values are truncated. By default, private names (leading underscore) are hidden; dunder methods such as ``__init__`` and
``__call__`` are always shown. Pass ``--private`` to include private helpers as well.
"""

import ast
import inspect
import sys
from typing import List, Optional, Tuple, Union

_Def = Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef]
_MAX_VALUE_WIDTH = 200


def _is_visible(name: str, include_private: bool) -> bool:
    is_dunder = name.startswith("__") and name.endswith("__")
    return include_private or is_dunder or not name.startswith("_")


def _header(node: _Def) -> str:
    if isinstance(node, ast.ClassDef):
        bases = ", ".join(ast.unparse(b) for b in node.bases)
        return f"class {node.name}({bases}):"
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    return f"{prefix} {node.name}({ast.unparse(node.args)}){returns}:"


def _assignment(node: ast.stmt) -> Optional[Tuple[str, str]]:
    """Return ``(name, rendered line)`` for a single-target assignment, else ``None``."""
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        value = f" = {ast.unparse(node.value)}" if node.value is not None else ""
        return node.target.id, f"{node.target.id}: {ast.unparse(node.annotation)}{value}"
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id, f"{node.targets[0].id} = {ast.unparse(node.value)}"
    return None


def _attribute_docstring(body: List[ast.stmt], index: int) -> Optional[str]:
    """Return the string literal directly following ``body[index]`` (an attribute docstring), if any."""
    if index + 1 < len(body):
        nxt = body[index + 1]
        if isinstance(nxt, ast.Expr) and isinstance(nxt.value, ast.Constant) and isinstance(nxt.value.value, str):
            return inspect.cleandoc(nxt.value.value)
    return None


def _show(parent: Union[ast.Module, ast.ClassDef], include_private: bool, indent: str = "") -> None:
    for index, node in enumerate(parent.body):
        assignment = _assignment(node)
        if assignment is not None:
            name, line = assignment
            if not _is_visible(name, include_private):
                continue
            if len(line) > _MAX_VALUE_WIDTH:
                line = line[: _MAX_VALUE_WIDTH - 3] + "..."
            print(indent + line)
            doc = _attribute_docstring(parent.body, index)
            if doc:
                body = doc.replace("\n", "\n" + indent)
                print(f'{indent}"""{body}"""')
            print()
            continue
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if not _is_visible(node.name, include_private):
            continue
        for decorator in node.decorator_list:
            print(f"{indent}@{ast.unparse(decorator)}")
        print(indent + _header(node))
        doc = ast.get_docstring(node)
        if doc:
            body = doc.replace("\n", "\n" + indent + "    ")
            print(f'{indent}    """{body}"""')
        else:
            print(f"{indent}    # NO DOCSTRING")
        if isinstance(node, ast.ClassDef):
            _show(node, include_private, indent + "    ")
        print()


def main(argv: list) -> int:
    include_private = "--private" in argv
    paths = [a for a in argv if a != "--private"]
    if not paths:
        print(__doc__)
        return 1
    for path in paths:
        with open(path, encoding="utf-8") as f:
            module = ast.parse(f.read(), filename=path)
        print(f"# ===== {path} =====")
        doc = ast.get_docstring(module)
        if doc:
            print(f'"""{doc}"""\n')
        _show(module, include_private)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
