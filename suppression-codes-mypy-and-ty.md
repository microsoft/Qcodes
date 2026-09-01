# Combining mypy and ty suppression codes

## Summary

The [ty suppression docs](https://docs.astral.sh/ty/suppression/) document putting
a ty rule into a mypy `type: ignore` comment by prefixing it with `ty:`:

```python
sum_three_numbers("one", 5, 2)  # type: ignore[arg-type, ty:invalid-argument-type]
```

ty honours this. **mypy does not ignore the `ty:` prefixed code**, and reports it
as an unused suppression when `warn_unused_ignores` is enabled, which qcodes
enables in `pyproject.toml`. So the combined form cannot be used here.

qcodes therefore uses two comments on the same line:

```python
f("one")  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
```

That is the only form of the three below that all three checkers accept.

## Results

| form | ty 0.0.74 | mypy 2.3.1 with `warn_unused_ignores` | mypy 2.3.1 without it | pyright 1.1.411 |
| --- | --- | --- | --- | --- |
| `# type: ignore[arg-type, ty:invalid-argument-type]` | suppressed | `Unused "type: ignore[ty:invalid-argument-type]" comment` | clean | suppressed |
| `# type: ignore[arg-type]` + `# ty: ignore[invalid-argument-type]` | suppressed | clean | clean | suppressed |
| `# type: ignore[ty:invalid-argument-type]` | suppressed | unused, and `arg-type` not covered | `arg-type` not covered | suppressed |

Note that the `arg-type` half of the combined form *is* honoured by mypy. It is
only the `ty:` prefixed code that mypy does not recognise, and therefore reports
as unused.

## How pyright fits in

pyright has its own suppression comment and also honours mypy's, which is why it
accepts all three forms above.

| comment | pyright |
| --- | --- |
| `# type: ignore` | suppressed |
| `# type: ignore[arg-type]` | suppressed |
| `# type: ignore[arg-type, ty:invalid-argument-type]` | suppressed |
| `# pyright: ignore` | suppressed |
| `# pyright: ignore[reportArgumentType]` | suppressed |
| `# pyright: ignore[reportGeneralTypeIssues]` | **not** suppressed, wrong rule |
| `# ty: ignore[invalid-argument-type]` | **not** suppressed |

Two things follow from this.

**`# type: ignore` is a blanket suppression for pyright.** pyright does not parse
the codes in it, so `# type: ignore[arg-type]` silences *every* pyright rule on
that line, not just the argument type one. A consequence that came up repeatedly
during the ty migration: removing a mypy suppression can surface a pyright error
on the same line that was never visible before. `# pyright: ignore[rule]` is the
precise form, and unlike `# type: ignore` it only suppresses the rules listed.

**A ty only suppression does not silence pyright.** `# ty: ignore[...]` is just a
comment as far as pyright is concerned. That is what makes the two comment form
safe: the mypy half keeps pyright quiet as a side effect, and the ty half is
inert for both of the others.

## Unused suppression detection

The three checkers differ in whether they tell you a suppression has gone stale.

| checker | setting | default | reports unused |
| --- | --- | --- | --- |
| mypy | `warn_unused_ignores` | off | enabled in `pyproject.toml` |
| ty | `unused-ignore-comment` | on | yes, for `ty: ignore` directives |
| pyright | `reportUnnecessaryTypeIgnoreComment` | off | not enabled, see below |

With the pyright setting enabled it reports all of these:

```python
def g(a: int) -> None: ...


g(1)  # type: ignore
g(1)  # pyright: ignore
g(1)  # pyright: ignore[reportArgumentType]
```

```
Unnecessary "# type: ignore" comment
Unnecessary "# type: ignore" comment
Unnecessary "# pyright: ignore" rule: "reportArgumentType"
```

**We cannot enable it while we also run mypy.** Because pyright treats
`# type: ignore` as a blanket suppression of *its own* rules, it calls the
comment unnecessary whenever pyright itself has nothing to report on the line,
with no knowledge of whether mypy needed it. Every mypy only suppression in the
code base would be reported as unnecessary. For example:

```python
from typing import Any


class A:
    def m(self) -> None: ...


def make(a: A, replacement: Any) -> None:
    # mypy reports method-assign here, pyright has no equivalent check
    a.m = replacement  # type: ignore[method-assign]
```

mypy needs that suppression: removing it gives
`error: Cannot assign to a method  [method-assign]`. pyright with
`reportUnnecessaryTypeIgnoreComment` enabled reports the very same line as
`Unnecessary "# type: ignore" comment`.

So mypy's `warn_unused_ignores` and ty's `unused-ignore-comment` are the two
stale suppression checks we can actually rely on.

## Test case

```python
def f(a: int) -> None: ...


# 1. combined form from the ty docs
f("one")  # type: ignore[arg-type, ty:invalid-argument-type]

# 2. the two comment form used in qcodes
f("one")  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

# 3. combined form, ty rule only
f("one")  # type: ignore[ty:invalid-argument-type]

# 4. control, expected to be reported by every checker
f("one")
```

And for the pyright specific forms:

```python
def h(a: int) -> None: ...


# 5. pyright: ignore, blanket
h("one")  # pyright: ignore

# 6. pyright: ignore with the matching rule
h("one")  # pyright: ignore[reportArgumentType]

# 7. pyright: ignore with a non matching rule
h("one")  # pyright: ignore[reportGeneralTypeIssues]

# 8. ty: ignore only
h("one")  # ty: ignore[invalid-argument-type]
```

Run from the repository root so that the mypy configuration in `pyproject.toml`
is picked up:

```
uv run ty check --output-format concise <file>
uv run --extra test mypy <file>
uv run --extra test mypy --no-warn-unused-ignores <file>
uv run pyright <file>
```

Expected results:

| block | ty | mypy | pyright |
| --- | --- | --- | --- |
| first, cases 1 to 4 | 4 | 1, 3, 4 and two unused directives | 4 |
| second, cases 5 to 8 | 5, 6, 7 | 5, 6, 7, 8 | 7, 8 |

The second block deliberately exercises comments that only one checker
understands, so most cases are reported by the other two. That is the point: it
shows that `pyright: ignore` is inert for mypy and ty, and that `ty: ignore` is
inert for mypy and pyright.

Any deviation from this table tells you that one of the checkers has changed how
it reads these comments.

## Why we keep `warn_unused_ignores`

Dropping `warn_unused_ignores` would make the combined form work, but that
setting is worth more than the shorter comments. As shown above it is, together
with ty's `unused-ignore-comment`, one of only two stale suppression checks
available to us. During the ty migration it caught:

- the `issuperset` suppression becoming redundant once
  [astral-sh/ty#4303](https://github.com/astral-sh/ty/issues/4303) was fixed in
  ty 0.0.74
- the two suppressions in the Keithley 7510 buffer becoming unnecessary once the
  data dictionary was annotated
- several suppressions in `ParameterBase` becoming unnecessary once the duck
  typed conversions were moved behind helpers

## Suggested upstream change

mypy could ignore codes carrying a `<tool>:` prefix in `type: ignore` comments,
rather than treating them as mypy codes that turned out to be unused. That would
make the form documented by ty usable in projects that run both checkers with
`warn_unused_ignores` enabled, and would generalise to any other checker that
wants to share the comment.

Failing that, the ty documentation could note that the combined form conflicts
with mypy's `warn_unused_ignores`, and suggest the two comment form for projects
that run both.

## Versions

Measured with ty 0.0.74, mypy 2.3.1 and pyright 1.1.411.
