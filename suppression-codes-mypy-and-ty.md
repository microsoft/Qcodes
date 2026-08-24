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

pyright honours a `# type: ignore` comment regardless of the codes in it, so it
accepts all three forms. That is also why removing a mypy suppression can
surface a pyright error on the same line.

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

Run from the repository root so that the mypy configuration in `pyproject.toml`
is picked up:

```
uv run ty check --output-format concise <file>
uv run --extra test mypy <file>
uv run --extra test mypy --no-warn-unused-ignores <file>
uv run pyright <file>
```

Only case 4 should be reported. Every checker reporting anything on cases 1 to 3
tells you which form is currently supported.

## Why we keep `warn_unused_ignores`

Dropping `warn_unused_ignores` would make the combined form work, but that
setting is worth more than the shorter comments. It is what tells us when a
suppression has become obsolete. During the ty migration it caught:

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
