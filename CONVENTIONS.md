# Opytimark conventions

Opytimark adopts the applicable cpmux/phitrain code-style rules. These conventions
apply to handwritten project code, not generated documentation, build output, or
installed dependencies.

## Compatibility and scope

- Preserve benchmark classes, constructor call forms, validated mutable metadata,
  exception interfaces, numerical behavior, examples, and bundled CEC data.
- Keep Python 3.11 support. The requested union syntax, builtin generics, and
  `collections.abc` imports already work on 3.11; do not introduce 3.12-only syntax
  without an explicit support-policy change.
- Keep Opytimark's Apache-2.0 license and existing public package exports. cpmux's
  MIT header, empty-package-initializer rule, application lifecycle, and runtime
  dependencies are not transferred to this library.
- Do not narrow historical array-shape or argument behavior as a formatting change.
  Describe and test intentional behavior changes separately.

## Code style

- Begin every `.py` file with the two-line copyright/license header:

  ```python
  # Copyright (c) 2020-2026 Gustavo de Rosa.
  # Licensed under the Apache License, Version 2.0.
  ```

- Use `X | None`, not `Optional[X]`, and builtin generics such as `list[str]` and
  `dict[str, Any]`. Import typing-specific constructs from `typing` and ABCs such as
  `Callable` and `Iterable` from `collections.abc`. (R2)
- Use top-level, absolute imports, ordered as standard library, third-party, and
  local imports, with a blank line between groups.
- Public functions, classes, explicit public constructors, and public property
  accessors have Google-style docstrings. Start with a single-sentence summary and
  keep each `Args:`, `Returns:`, and `Raises:` entry on one line. Do not add
  semicolons or `defaults to ...` tails to entries. (R3, R13)
- Document constructor arguments on `__init__`, not on the regular class.
  Benchmark class `Notes:` retain their equations, domains, and optimum
  qualifications after the one-line summary. `__call__` is a public scientific API,
  not a framework-only hook.
- Leave one blank line before a docstring's closing `"""` and one blank line
  between the closing delimiter and the next statement or field.
- Private helpers and private implementation classes have no docstrings.
  Framework-only overrides have none unless they are also a documented public API.
- Module docstrings are optional. Omit redundant module-title boilerplate rather
  than adding filler solely to satisfy formatting.
- Data classes with generated constructors document every field in an
  `Attributes:` section, using one `name: description.` entry per field.
- Use `get_logger(__name__)` from `opytimark.logging`, never `print()` in library
  code. The library does not configure application log levels or output streams.
  Example scripts may print their results.
- Diagnostic warnings and errors have a backticked offender and trailing period,
  such as `` f"`name=value` could not be loaded." ``. Informational and debug
  messages use ordinary prose. (R14)
- Raised messages identify the offender in backticks, use a verb phrase, and end
  with a period. Use `is None` and `is True` prose where applicable. (R1)
- Runtime validation uses `if` and a specific raised exception, never `assert`.
  Bare `except:` is forbidden.
- Comments explain why, not what. Prefer none or one line, allow at most three
  consecutive lines, and avoid banners or trailing periods. The required legal
  header is exempt from the comment-punctuation rule. (R8)
- Use one blank line at phase transitions in function bodies of at least 12 lines.
  Separate validation from mutation and preparation from evaluation, but do not
  mechanically separate related assignments. (R11)
- Inline first. Extract a helper, constant, or parameter when a second call site
  demonstrates the shared responsibility. (R16)
- Use double-quoted strings and a 120-character formatting/prose limit.
  Mathematical notation and reference URLs are not reworded merely to meet a
  prose limit. (R9)

## Numerical contracts

- Use `ArrayLike` at conversion boundaries and `NDArray` for normalized inputs.
  Preserve existing scalar/array return types and precision rather than forcing
  `float64` conversions to satisfy annotations.
- Keep raw strings for docstrings containing LaTeX. Mathematical commands use a
  single backslash inside a raw string.
- Preserve callback order, random draws, input ownership, and component-instance
  independence when sharing numerical implementations.

## Tests and tooling

- Keep the existing test layout. Test functions remain plain functions without
  docstrings or type annotations, following cpmux's test convention.
- Use ordinary pytest assertions without failure-message strings. Runtime
  validation and test assertions serve different purposes. (R15)
- Use behavior-oriented names and parameterization where cases share a contract.
  Do not manufacture universal finite-output or determinism expectations.
- Keep Black, isort, and Flake8, with a consistent line length of 120.
  Convention coverage, numerical regressions, and strict Sphinx builds complement
  formatting rather than treating formatting as proof of correctness.
