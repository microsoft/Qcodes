---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Complex Numbers

QCoDeS natively supports complex numbers via the complex datatypes of `numpy`.

```{code-cell} ipython3
import numpy as np
```

## Complex-valued parameters

QCoDeS parameters can take complex values. There are two types of complex-valued parameters: scalar-valued parameters and array-valued parameters.

+++

### Scalar-valued parameters

Let us create a complex-valued parameter and `set` and `get` values for it. An example that one might encounter in physics is the complex impedance.

For any QCoDeS parameter, it holds that adding **input validation** is a good idea. Complex parameters are no exception. We therefore use the `ComplexNumbers` validator with our complex parameter.

```{code-cell} ipython3
from qcodes.instrument.parameter import Parameter
from qcodes.utils.validators import ComplexNumbers
```

```{code-cell} ipython3
imp = Parameter(name='imp',
                label='Impedance',
                unit='Ohm',
                initial_value=50+0j,
                set_cmd=None,
                get_cmd=None,
                vals=ComplexNumbers())
```

The `ComplexNumbers` validator requires explicitly complex values. `float`s and `int`s will *not* pass.

```{code-cell} ipython3
for value in np.array([1, -1, 8.2, np.pi]):
    try:
        imp(value)
        print(f'Succesfully set the parameter to {value}')
    except TypeError:
        print(f'Sorry, but {value} is not complex')
```

The easiest way to make a scalar value complex is probably by adding `0j` to it.

```{code-cell} ipython3
for value in np.array([1, -1, 8.2, np.pi]) + 0j:
    try:
        imp(value)
        print(f'Succesfully set the parameter to {value}')
    except TypeError:
        print(f'Sorry, but {value} is not complex')
```

### Array-valued parameters

There is no separate complex-valued array validator, since the `Arrays` validator can be customized to cover any real or complex valued case.

Let's make a little array to hold some quantum state amplitudes. Let's pretend to be in a 5-dimensional Hilbert space. Our state parameter should thus hold 5 complex numbers (the state expansion coefficients in some implicit basis).

```{code-cell} ipython3
from qcodes.utils.validators import Arrays
```

The proper validator should accept complex numbers and should reject anything of the wrong shape. Note that we get to decide whether we want to accept "non-strictly complex" data.

```{code-cell} ipython3
amps_val_strict = Arrays(shape=(5,), valid_types=(np.complex,))
amps_val_lax = Arrays(shape=(5,), valid_types=(np.complex, np.float, np.int))
```

The strict validator is strict:

```{code-cell} ipython3
try:
    amps_val_strict.validate(np.array([1, 2, 3, 4, 5]))
except TypeError:
    print('Sorry, but integers are not strictly complex')
    
try:
    amps_val_strict.validate(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
except TypeError:
    print('Sorry, but floats are not strictly complex')
```

But note, that the presence of a single imaginary part will cast the whole array as complex:

```{code-cell} ipython3
my_array = np.array([1.0 + 0j, 2.0, 3.0, 4.0, 5.0])
print(my_array)
print(my_array.dtype)
amps_val_strict.validate(np.array([1.0 + 0j, 2.0, 3.0, 4.0, 5.0]))
print('Yeah, those are complex numbers')
```

The lax validator let's everything through:

```{code-cell} ipython3
amps_val_lax.validate(np.array([1, 2, 3, 4, 5]))
amps_val_lax.validate(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
amps_val_lax.validate(np.array([1.0 + 0j, 2, 3, 4, 5]))
```

We can use either validator for the parameter.

```{code-cell} ipython3
amplitudes = Parameter(name='amplitudes',
                       label='Amplitudes',
                       unit='',
                       set_cmd=None,
                       get_cmd=None,
                       vals=amps_val_strict,
                       initial_value=(1/np.sqrt(2)*np.array([1+1j, 0, 0, 0, 0])))
```

```{code-cell} ipython3
amplitudes()
```

```{code-cell} ipython3
amplitudes(1/np.sqrt(2)*np.array([0, 1+1j, 0, 0, 0]))
```

```{code-cell} ipython3

```
