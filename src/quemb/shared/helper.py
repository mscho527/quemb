from collections.abc import Callable, Iterable
from inspect import signature
from itertools import islice
from pathlib import Path
from time import time
from typing import Any, TypeVar, overload

import numba as nb
from attr import define, field

from quemb.shared.typing import Integral, T

_Function = TypeVar("_Function", bound=Callable)
_T_Integral = TypeVar("_T_Integral", bound=Integral)


# Note that we have once Callable and once Function.
# This is **intentional**.
# The inner function `update_doc` takes a function
# and returns a function **with** the exact same signature.
def add_docstring(doc: str | None) -> Callable[[_Function], _Function]:
    """Add a docstring to a function as decorator.

    Is useful for programmatically generating docstrings.

    Parameters
    ----------
    doc: str
        A docstring.

    Example
    ----------
    >>> @add_docstring("Returns 'asdf'")
    >>> def f():
    >>>     return 'asdf'
    is equivalent to
    >>> def f():
    >>>     "Returns 'asdf'"
    >>>     return 'asdf'
    """

    def update_doc(f: _Function) -> _Function:
        f.__doc__ = doc
        return f

    return update_doc


def ensure(condition: bool, message: str = "") -> None:
    """This function can be used instead of :python:`assert`,
    if the test should be always executed.
    """
    if not condition:
        message = message if message else "Invariant condition was violated."
        raise ValueError(message)


def copy_docstring(f: Callable) -> Callable[[_Function], _Function]:
    """Copy docstring from another function as decorator."""
    return add_docstring(f.__doc__)


def _get_init_docstring(obj: type) -> str:
    sig = signature(obj.__init__)  # type: ignore[misc]
    docstring = """Initialization

Parameters
----------
"""
    # we want to skip `self`
    for var in islice(sig.parameters.values(), 1, None):
        docstring += f"{var.name}: {var.annotation}\n"
    return docstring


def add_init_docstring(obj: type) -> type:
    """Add a sensible docstring to the __init__ method of an attrs class

    Makes only sense if the attributes are type-annotated.
    Is a stopgap measure until https://github.com/sphinx-doc/sphinx/issues/10682
    is solved.
    """
    obj.__init__.__doc__ = _get_init_docstring(obj)  # type: ignore[misc]
    return obj


def unused(*args: Any) -> None:
    pass


def ncore_(z: int) -> int:
    if 1 <= z <= 2:
        nc = 0
    elif 2 <= z <= 5:
        nc = 1
    elif 5 <= z <= 12:
        nc = 1
    elif 12 <= z <= 30:
        nc = 5
    elif 31 <= z <= 38:
        nc = 9
    elif 39 <= z <= 48:
        nc = 14
    elif 49 <= z <= 56:
        nc = 18
    else:
        raise ValueError("Ncore not computed in helper.ncore(), add it yourself!")
    return nc


def delete_multiple_files(*args: Iterable[Path]) -> None:
    for files in args:
        for file in files:
            file.unlink()


@define(frozen=True)
class Timer:
    """Simple class to time code execution"""

    message: str = "Elapsed time"
    start: float = field(init=False, factory=time)

    def elapsed(self) -> float:
        return time() - self.start

    def str_elapsed(self, message: str | None = None) -> str:
        return f"{self.message if message is None else message}: {self.elapsed():.5f}"


@overload
def njit(f: _Function) -> _Function: ...
@overload
def njit(**kwargs: Any) -> Callable[[_Function], _Function]: ...


def njit(
    f: _Function | None = None, **kwargs: Any
) -> _Function | Callable[[_Function], _Function]:
    """Type-safe jit wrapper that caches the compiled function

    With this jit wrapper, you can actually use static typing together with numba.
    The crucial declaration is that the decorated function's interface is preserved,
    i.e. mapping :class:`Function` to :class:`Function`.
    Otherwise the following example would not raise a type error:

    .. code-block:: python

        @numba.njit
        def f(x: int) -> int:
            return x

        f(2.0)   # No type error

    While the same example, using this custom :func:`njit` would raise a type error.

    In addition to type safety, this wrapper also sets :code:`cache=True` by default.
    """
    if f is None:
        return nb.njit(cache=True, **kwargs)
    else:
        return nb.njit(f, cache=True, **kwargs)


@overload
def jitclass(cls_or_spec: T, spec: list[tuple[str, Any]] | None = ...) -> T: ...


@overload
def jitclass(
    cls_or_spec: list[tuple[str, Any]] | None = None, spec: None = None
) -> Callable[[T], T]: ...


def jitclass(
    cls_or_spec: T | list[tuple[str, Any]] | None = None,
    spec: list[tuple[str, Any]] | None = None,
) -> T | Callable[[T], T]:
    """Decorator to make a class jit-able.

    The rationale is the same as for :func:`njit`, and described there.

    For a more detailed explanation of numba jitclasses,
    see https://numba.readthedocs.io/en/stable/user/jitclass.html
    """
    return nb.experimental.jitclass(cls_or_spec, spec)


@njit
def gauss_sum(n: _T_Integral) -> _T_Integral:
    r"""Return the sum :math:`\sum_{i=1}^n i`

    Parameters
    ----------
    n :
    """
    return (n * (n + 1)) // 2  # type: ignore[return-value]


@njit
def ravel_symmetric(a: _T_Integral, b: _T_Integral) -> _T_Integral:
    """Flatten the index a, b assuming symmetry.

    The resulting indexation for a matrix looks like this::

        0
        1   2
        3   4   5
        6   7   8   9

    Parameters
    ----------
    a :
    b :
    """
    return gauss_sum(a) + b if a > b else gauss_sum(b) + a  # type: ignore[return-value,operator]


@njit
def ravel(a: _T_Integral, b: _T_Integral, n_cols: _T_Integral) -> _T_Integral:
    """Flatten the index a, b assuming row-mayor/C indexing

    The resulting indexation for a 3 by 4 matrix looks like this::

        0   1   2   3
        4   5   6   7
        8   9  10  11


    Parameters
    ----------
    a :
    b :
    n_rows :
    """
    assert b < n_cols  # type: ignore[operator]
    return (a * n_cols) + b  # type: ignore[return-value,operator]


@njit
def symmetric_different_size(m: _T_Integral, n: _T_Integral) -> _T_Integral:
    r"""Return the number of unique elements in a symmetric matrix of different row
    and column length

    This is for example the situation for pairs :math:`\mu, i` where :math:`\mu`
    is an AO and :math:`i` is a fragment orbital.

    The assumed structure of the symmetric matrix is::

        *   *   *   *
        *   *   *   *
        *   *   0   0
        *   *   0   0

    where the stars denote non-zero elements.

    Parameters
    ----------
    m:
    n:
    """

    m, n = min(m, n), max(m, n)  # type: ignore[type-var]
    return gauss_sum(m) + m * (n - m)  # type: ignore[operator,return-value]


@njit
def get_flexible_n_eri(
    p_max: _T_Integral, q_max: _T_Integral, r_max: _T_Integral, s_max: _T_Integral
) -> _T_Integral:
    r"""Return the number of unique ERIs but allowing different number of orbitals.

    This is for example the situation for a tuple :math:`\mu, \nu, \kappa, i`,
    where :math:`\mu, \nu, \kappa` are AOs and :math:`i` is a fragment orbital.
    This function returns the number of unique ERIs :math:`g_{\mu, \nu, \kappa, i}`.

    Parameters
    ----------
    p_max:
    q_max:
    r_max:
    s_max:
    """

    return symmetric_different_size(
        symmetric_different_size(p_max, q_max), symmetric_different_size(r_max, s_max)
    )
