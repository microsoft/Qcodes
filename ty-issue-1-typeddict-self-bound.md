# ty issue draft 1

**Title**

> Generic `TypedDict` with a type parameter default: `Self`-bound methods and `**`-unpacking rejected for every non-default specialization

**Labels to suggest:** `bug`, `generics`, `typeddict`, `constraint-solver`

---

### Summary

When a generic `TypedDict` declares a default for its type parameter, ty computes
the upper bound of the synthesized `Self` type variable as the *default*
specialization rather than the generic one. Every other specialization is then
rejected by any method that binds `Self`.

```python
from typing import TypedDict


class Movie[T = int](TypedDict):
    extra: T


def f(m: Movie[str]) -> None:
    m.keys()
```

```
error[invalid-argument-type]: Argument to bound method `TypedDictFallback.keys` is incorrect
 --> repro.py:7:5
  |
7 |     m.keys()
  |     ^^^^^^^^ Argument type `Movie[str]` does not satisfy upper bound `Movie[int]` of type variable `Self`
```

Removing the default (`class Movie[T](TypedDict)`) makes the error go away
without any other change, so the default is what introduces the bound.

Note that `Movie[str]` here is an ordinary concrete specialization. No type
variable is unsolved at the call site, and nothing is being inferred.

### Which specializations are affected

Only the declared default is accepted:

| annotation | result |
| --- | --- |
| `Movie[int]` (the default) | ok |
| `Movie` (bare, default applies) | ok |
| `Movie[str]` | error |
| `Movie[T]` for an enclosing type variable `T` | error |

### Which members are affected

Members whose signature binds `Self`:

| member | result |
| --- | --- |
| `keys()` | error |
| `values()` | error |
| `items()` | error |
| `copy()` | error |
| `**` unpacking | error |
| `get()` | ok |
| `setdefault()` | ok |
| `pop()` | ok |
| `update()` | ok |

Assignability is unaffected, which is consistent with the problem being the
`Self` bound rather than the type itself:

```python
from typing import Mapping, TypedDict


class Movie[T = int](TypedDict):
    extra: T


def f(m: Movie[str]) -> None:
    ok: Mapping[str, object] = m  # no error
```

### The `**` unpacking symptom

`**`-unpacking reports a different and rather misleading message, which is how I
originally ran into this:

```python
from typing import TypedDict


class Movie[T = int](TypedDict):
    extra: T


def f(m: Movie[str]) -> None:
    dict(**m)
```

```
error[invalid-argument-type]: Argument expression after ** must be a mapping type
 --> repro.py:7:12
  |
7 |     dict(**m)
  |            ^ Found `Movie[str]`
```

A `TypedDict` is always a `Mapping[str, object]`, so this message points away
from the real cause.

### Not specific to `TypedDict` syntax or version

The legacy spelling behaves identically:

```python
from typing import Generic, TypedDict, TypeVar

T = TypeVar("T", default=int)


class Movie(TypedDict, Generic[T]):
    extra: T


def f(m: Movie[str]) -> None:
    m.copy()
```

A plain generic class with a type parameter default is **not** affected, so this
looks specific to the synthesized `TypedDictFallback` `Self`:

```python
class WithDefault[T = int]:
    def m(self) -> None: ...


def f[T](a: WithDefault[T]) -> None:
    a.m()  # no error
```

Any non-`Any` default triggers it. `Any` is the only default that is accepted,
which is probably why this has gone unnoticed:

| type parameter | result |
| --- | --- |
| `class Movie[T](TypedDict)` | ok |
| `class Movie[T = Any](TypedDict)` | ok |
| `class Movie[T = int](TypedDict)` | error |
| `class Movie[T = None](TypedDict)` | error |
| `class Movie[T = int \| None](TypedDict)` | error |
| `class Movie[T = object](TypedDict)` | error |
| `class Movie[T: int \| None = int \| None](TypedDict)` | error |

Reproduced on 0.0.72, 0.0.73 and 0.0.74. Checked with `--python-version 3.13` so
that the PEP 696 syntax is not itself reported as an error. mypy 2.3.1 and
pyright both accept all of the above.

I searched existing issues for `"must be a mapping type"`, `TypedDict Unpack
default`, `"generic TypedDict"`, `"PEP 696"` and `Unpack kwargs` and did not
find a preexisting issue. #4255 is the closest but is about a union alias in a
stub leaking an unspecialized type variable.

The error shape is reminiscent of #4303, which is also an upper bound on a type
variable being applied too strictly, though that one is about a `bound=` on a
class scoped type variable rather than a `default=` on `Self`.

### Relation to the feature overview

The type system feature overview in #1889 lists all of the following as
implemented:

- Generics: `TypeVar` defaults (PEP 696)
- `TypedDict`: Inheritance, generic `TypedDict`s
- `TypedDict`: Structural assignability and equivalence
- `TypedDict`: Methods (`get`, `pop`, `setdefault`, `keys`, `values`, `copy`)

This report sits at the intersection of those, so following the guidance at the
top of #1889 for features marked completed, it seemed worth reporting rather
than upvoting a tracking issue.

It is worth stressing that this is **not** about `Unpack` for `**kwargs` typing,
which #1889 tracks separately in #1746. The lead repro contains no `Unpack` and
no `**` at all, just `Movie[str].keys()`. The `**` message is only how I
happened to notice it.

Structural assignability also still works (`Mapping[str, object] = m` is
accepted), so this looks narrowly scoped to the upper bound computed for the
synthesized `Self`.

### Version

0.0.74
