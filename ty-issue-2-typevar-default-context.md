# ty issue draft 2

**Title**

> Function scoped `TypeVar` default takes precedence over the declared type context, where an unsolved type variable would be accepted

**Labels to suggest:** `bidirectional inference`, `constraint-solver`, `generics`

---

### Summary

When a function scoped type variable appears only in the return type and is not
constrained by any argument, ty leaves it unsolved as `Unknown`, which is
gradually compatible with whatever the result is assigned to. If that same type
variable declares a PEP 696 default, ty substitutes the default instead, which
is concrete and then conflicts with the declared type.

```python
class Box[T]:
    pass


def make[T = int](cls: type[T] | None = None) -> Box[T]:
    raise NotImplementedError


def caller() -> None:
    a: Box[str] = make()
```

```
error[invalid-assignment]: Object of type `Box[int]` is not assignable to `Box[str]`
 --> repro.py:8:19
  |
8 |     a: Box[str] = make()
  |                   ^^^^^^
```

Removing the default makes ty accept it:

```python
class Box[T]:
    pass


def make[T](cls: type[T] | None = None) -> Box[T]:
    raise NotImplementedError


def caller() -> None:
    a: Box[str] = make()  # ty: ok
```

`reveal_type` shows what is actually happening. The declared type is never used
to solve `T` in either case; the difference is only what fills the unsolved slot:

| declaration | `reveal_type(make())` | `a: Box[str] = make()` |
| --- | --- | --- |
| `def make[T](...) -> Box[T]` | `Box[Unknown]` | accepted |
| `def make[T = int](...) -> Box[T]` | `Box[int]` | error |

So adding a default is strictly worse than having no default at all, at every
call site that annotates its target. mypy 2.3.1 and pyright accept both forms.

### The type context is available

This is not a case of ty lacking the necessary context. Using the example from
#3933, the declared type of the assignment target clearly does reach the
constraint solver, since it widens the argument:

```python
class Parent: ...


class Child(Parent): ...


def head[T](x: list[T]) -> T:
    return x[0]


x: Parent = head(reveal_type([Child()]))  # revealed: list[Parent]
```

I reproduced that on 0.0.74. So in `a: Box[str] = make()` the constraint
`Box[T] <: Box[str]` is available, but the default is applied in preference to
it.

### Why this matters

This pattern is common in factory functions, where the default exists to give a
sensible type to an unannotated call while still allowing the caller to ask for
something more specific (illustrative, from our codebase):

```python
p = instrument.add_parameter("name")  # want the default
q: Parameter[float, Self] = instrument.add_parameter("x")  # want this instead
```

With ty's current behaviour the default wins in both cases, so the second form
is unusable and every annotated call site becomes an error. In our codebase this
produced 34 errors across instrument drivers from a single type variable
declaration. We ended up widening the default to a fully gradual type to work
around it, which loses the information the default was there to provide.

### Relation to #3933 and the feature overview

This looks like it may fall under #3933, constraint-set-aware bidirectional
inference. That issue is written in terms of constraints flowing into *argument*
inference, and all of its examples involve arguments that get eagerly
specialized or wrongly widened. The case here has no arguments at all, so the
symptom is different, but the underlying gap looks similar: the outer constraint
is not being unified with the specialization of the call.

If the second approach in #3933 is taken, propagating constraints during
bidirectional inference rather than eagerly specializing, then `Box[T] <:
Box[str]` should presumably solve `T` to `str` before any default is considered,
which would fix this too. Filing separately in case that is not the intent, and
because the interaction with PEP 696 defaults is not mentioned there.

The type system feature overview in #1889 lists "`TypeVar` defaults (PEP 696)"
as implemented under Generics. That section also has an open sub-item, "Solve
type variables in all cases" (#623), which may be the more appropriate home if
this is considered a solver limitation rather than a deliberate choice about
defaults.

### Note on the spec

I could not find wording in PEP 696 or the typing spec that settles whether the
declared type context should take precedence over a type variable default, so
this may be intentional. If it is, it would be helpful to say so explicitly,
since the natural reading of "the default is used when the type variable cannot
be solved" is that a solution derived from the type context counts as solving
it. The current behaviour also has the surprising property that adding a default
makes a call site fail that would otherwise have been accepted.

Reproduced on 0.0.72, 0.0.73 and 0.0.74, checked with `--python-version 3.13`.

### Version

0.0.74
