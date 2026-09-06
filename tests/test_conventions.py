# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import ast
import io
import re
import tokenize
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = [
    "# Copyright (c) 2020-2026 Gustavo de Rosa.",
    "# Licensed under the Apache License, Version 2.0.",
]


@cache
def _sources():
    paths = {path for directory in ("opytimark", "examples", "tests") for path in (ROOT / directory).rglob("*.py")}
    paths.add(ROOT / "docs" / "conf.py")

    return [
        (path.relative_to(ROOT), source, ast.parse(source, filename=str(path)))
        for path in sorted(paths)
        for source in [path.read_text(encoding="utf-8")]
    ]


def _declarations(tree):
    pending = [(tree, False)]
    while pending:
        node, private_context = pending.pop()
        for child in ast.iter_child_nodes(node):
            private = private_context
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                private |= child.name.startswith("_") and child.name not in {
                    "__init__",
                    "__call__",
                }
                yield child, private
            nested = private or isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            pending.append((child, nested))


def test_python_files_have_project_license_headers():
    violations = [str(path) for path, source, _ in _sources() if source.splitlines()[:2] != HEADER]

    assert not violations


def test_library_uses_absolute_top_level_imports_and_modern_types():
    prohibited = {
        "Optional",
        "Union",
        "List",
        "Dict",
        "Tuple",
        "Set",
        "FrozenSet",
        "Type",
        "Callable",
        "Iterable",
        "Iterator",
        "Mapping",
        "MutableMapping",
        "Sequence",
        "Collection",
        "Generator",
        "Awaitable",
        "Coroutine",
    }
    violations = []
    for path, _, tree in _sources():
        if path.parts[0] != "opytimark":
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    violations.append((str(path), node.lineno, "relative import"))
                if node.module == "typing":
                    for alias in node.names:
                        if alias.name in prohibited:
                            violations.append((str(path), node.lineno, alias.name))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                for child in ast.walk(node):
                    if isinstance(child, (ast.Import, ast.ImportFrom)):
                        violations.append((str(path), child.lineno, "nested import"))
            annotations = []
            if isinstance(node, ast.arg) and node.annotation is not None:
                annotations.append(node.annotation)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.returns is not None:
                annotations.append(node.returns)
            for annotation in annotations:
                if ast.unparse(annotation) == "np.array":
                    violations.append((str(path), annotation.lineno, "np.array annotation"))

    assert not violations


def test_library_documents_public_apis_not_private_helpers():
    violations = []
    for path, _, tree in _sources():
        if path.parts[0] != "opytimark":
            continue
        for node, private in _declarations(tree):
            doc = ast.get_docstring(node)
            if private:
                if doc is not None:
                    violations.append((str(path), node.lineno, "private docstring"))
                continue
            if doc is None:
                violations.append((str(path), node.lineno, "missing public docstring"))
                continue

            summary = doc.splitlines()[0]
            if not summary.endswith(".") or len(summary) > 120:
                violations.append((str(path), node.lineno, "summary"))
            if isinstance(node, ast.ClassDef):
                if "Args:" in doc.splitlines():
                    violations.append((str(path), node.lineno, "constructor Args on class"))
                continue

            arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
            names = [argument.arg for argument in arguments]
            if names and names[0] in {"self", "cls"}:
                names.pop(0)
            names.extend(argument.arg for argument in (node.args.vararg, node.args.kwarg) if argument is not None)
            documented = set(re.findall(r"(?m)^    ([A-Za-z_]\w*):", doc))
            for missing in set(names) - documented:
                violations.append((str(path), node.lineno, f"undocumented {missing}"))

    assert not violations


def test_library_docstrings_keep_closing_and_body_spacing():
    violations = []
    for path, source, tree in _sources():
        if path.parts[0] != "opytimark":
            continue
        lines = source.splitlines()
        containers = [tree, *(node for node, _ in _declarations(tree))]
        for node in containers:
            if ast.get_docstring(node) is None:
                continue
            doc = node.body[0]
            if doc.end_lineno <= doc.lineno or lines[doc.end_lineno - 2].strip():
                violations.append((str(path), doc.lineno, "blank before closing delimiter"))
            if doc.end_lineno - doc.lineno >= 2 and not lines[doc.end_lineno - 3].strip():
                violations.append((str(path), doc.lineno, "multiple closing blank lines"))
            if len(node.body) < 2:
                continue
            next_content = doc.end_lineno
            while next_content < len(lines) and not lines[next_content].strip():
                next_content += 1
            if next_content - doc.end_lineno != 1:
                violations.append((str(path), doc.lineno, "blank after closing delimiter"))

    assert not violations


def test_library_docstring_entries_are_single_line_google_style():
    violations = []
    sections = {"Args:", "Returns:", "Raises:", "Attributes:"}
    for path, _, tree in _sources():
        if path.parts[0] != "opytimark":
            continue
        for node, private in _declarations(tree):
            doc = ast.get_docstring(node)
            if private or doc is None:
                continue
            section = None
            return_lines = 0
            for line in doc.splitlines():
                if line and not line.startswith(" "):
                    section = line if line in sections else None
                    return_lines = 0
                    continue
                if not section or not line.strip():
                    continue
                if ";" in line or re.search(r"defaults to\b", line, re.IGNORECASE):
                    violations.append((str(path), node.lineno, "entry wording"))
                if section == "Returns:":
                    return_lines += 1
                    if return_lines > 1 or not line.startswith("    "):
                        violations.append((str(path), node.lineno, "multiline Returns entry"))
                elif not re.match(r"^    [A-Za-z_][\w.]*:\s+\S", line):
                    violations.append((str(path), node.lineno, "multiline or malformed entry"))

    assert not violations


def test_library_math_commands_are_not_double_escaped():
    violations = []
    pattern = re.compile(r"\\\\(?:alpha|approx|ast|beta|frac|lambda|pi|sqrt|text|theta)(?![A-Za-z])")
    for path, _, tree in _sources():
        if path.parts[0] != "opytimark":
            continue
        for node, _ in _declarations(tree):
            doc = ast.get_docstring(node) or ""
            if pattern.search(doc):
                violations.append((str(path), node.lineno))

    assert not violations


def test_library_docstring_prose_fits_the_line_limit():
    violations = []
    for path, source, tree in _sources():
        if path.parts[0] != "opytimark":
            continue
        lines = source.splitlines()
        for node, _ in _declarations(tree):
            if ast.get_docstring(node) is None:
                continue
            doc = node.body[0]
            for number in range(doc.lineno, doc.end_lineno + 1):
                line = lines[number - 1]
                if len(line) <= 120 or any(marker in line for marker in (":math:", ".. math::", "https://")):
                    continue
                violations.append((str(path), number))

    assert not violations


def test_library_runtime_validation_does_not_print_assert_or_catch_everything():
    violations = []
    for path, _, tree in _sources():
        if path.parts[0] != "opytimark":
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                violations.append((str(path), node.lineno, "runtime assert"))
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                violations.append((str(path), node.lineno, "bare except"))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
                violations.append((str(path), node.lineno, "library print"))

    assert not violations


def test_explicit_library_error_messages_identify_the_offender():
    violations = []
    for path, _, tree in _sources():
        if path.parts[0] != "opytimark":
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Raise) or not isinstance(node.exc, ast.Call) or not node.exc.args:
                continue
            message = node.exc.args[0]
            if isinstance(message, ast.Constant) and isinstance(message.value, str):
                parts = [message.value]
            elif isinstance(message, ast.JoinedStr):
                parts = [part.value for part in message.values if isinstance(part, ast.Constant)]
            else:
                continue
            if not parts or not parts[0].startswith("`") or not parts[-1].endswith("."):
                violations.append((str(path), node.lineno))

    assert not violations


def test_comments_are_short_unpunctuated_and_without_banners():
    violations = []
    for path, source, _ in _sources():
        previous_line = -1
        length = 0
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type != tokenize.COMMENT or token.start[0] <= 2:
                continue
            length = length + 1 if token.start[0] == previous_line + 1 else 1
            previous_line = token.start[0]
            text = token.string[1:].strip()
            if length > 3 or text.endswith(".") or re.search(r"[-=#*]{4,}", text):
                violations.append((str(path), token.start[0], text))

    assert not violations
