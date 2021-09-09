from adaled.backends import TensorCollection, cmap, get_backend
from adaled.utils.dataclasses_ import DataclassMixin, dataclass
from adaled.utils.misc import batch_mse

import numpy as np
import torch

from typing import Callable, Optional, Sequence, Tuple, Union

_CollectionLike = Union[np.ndarray, torch.Tensor, TensorCollection]
_CmpErrorFunc = Callable[[_CollectionLike, _CollectionLike], _CollectionLike]

_int_like = (int, np.integer)
_tuple_like = (list, tuple)

class AdaLEDCriteria:
    """Base class for stopping criteria of different stages of an AdaLED cycle."""

    def should_continue_warmup(self, step, x, F, z, h, transformer) -> bool:
        """Return whether the warmup phase should continue or not.

        Arguments:
            step: current warmup step
            x: real state
            F: external forcing
            z: latent state (may be ``None`` in the first step)
            h: hidden state (may be ``None`` in the first step)
        """
        raise NotImplementedError()

    def should_end_comparison(self, step, x, F, z, h, transformer) -> bool:
        """Return whether the next step should be the final micro-macro
        comparison step.

        Note that the at least one comparison step has to be performed, in
        order to compute the uncertainty which is then passed to
        ``should_run_macro``.
        """
        raise NotImplementedError()

    def compute_error(self, x_macro, x_micro) -> Sequence[float]:
        """Compute the comparison error between macro and micro states."""
        raise NotImplementedError()

    def compute_and_check_error(self, uncertainty, x, F, z, h, transformer) \
            -> Tuple[Sequence[float], bool]:
        """Return the comparison errors between the micro and macro states
        according to any desired metric, and whether the errors and
        uncertainties are all small enough to use the macro solver.

        The result of this function is recorded for post-analysis. The
        implementation may decide to return ``np.nan`` if the computation of
        the error is undesired for performance reasons.

        Returns:
            (batch of errors, are all errors acceptable)
        """
        raise NotImplementedError()

    def should_accept_macro_step(self, step, uncertainty, F, z, h, transformer) -> bool:
        raise NotImplementedError()

    def should_continue_micro(self, step, x, F) -> bool:
        raise NotImplementedError()

    def should_continue_relaxation(self, step, x, F) -> bool:
        """Return whether the post-macro relaxation phase should continue.

        Arguments:
            step: current relaxation step
            x: real state
            F: external forcing
        """
        raise NotImplementedError()


@dataclass
class SimpleCriteriaConfig(DataclassMixin):
    """Configuration of SimpleCriteria.

    Attributes:
        k_warmup (int): number of steps for teacher-forcing warm-up
        k_cmp (int): number of steps used to evaluate macro solver's accuracy
        max_cmp_error (float): the macro solver is run only if error after
                      micro-macro comparison is below the given threshold,
                      where by default the MSE is used
        max_uncertainty (int): maximum acceptable macro uncertainty
        max_macro_steps (int or pair of ints): maximum number of macro steps to
                         perform (*)
        max_micro_steps (int or pair of ints): maximum number of steps in the
                        micro-only stage, does not include warm-up and
                        comparison stages (*)
        num_relaxation_steps (int): number of post-macro relaxation steps

    (*) If a pair of ints :math:`(a, b)` is given, the maximum number :math:`k`
    of corresponding step is sampled uniformly at random for each cycle, with
    :math:`a \leq k \leq b`.
    """
    k_warmup: int = 10
    k_cmp: int = 94
    max_cmp_error: float = 0.1
    max_uncertainty: float = 0.1
    max_macro_steps: Union[int, Tuple[int, int]] = 200
    max_micro_steps: Union[int, Tuple[int, int]] = 200
    num_relaxation_steps: int = 10

    def create(self):
        """Create the :class:`SimpleCriteria` object with this config."""
        return SimpleCriteria(self)

    def expected_max_utilization(self) -> float:
        """Compute the expected value of the maximum macro utilization
        (fraction of time steps performed in the macro-only stage)."""
        m = self.max_macro_steps
        if isinstance(m, int):
            m = np.int32(m)
        else:
            m = np.arange(m[0], m[1] + 1)
        return (m / (self.k_warmup + self.k_cmp + m + self.num_relaxation_steps)).mean()


class SimpleCriteria(AdaLEDCriteria):
    """Simple stopping criteria based on step index, uncertainty and MSE (or
    a custom comparison error function).

    Apart from computing the comparison error as the criterion for starting the
    macro solver, the states are otherwise ignored.
    """
    def __init__(self,
                 config: SimpleCriteriaConfig,
                 error_func: Optional[_CmpErrorFunc] = None,
                 seed: Optional[int] = None):
        """
        Arguments:
            config (SimpleCriteriaConfig):
            error_func (optional): Function of two arguments (x_macro, x_micro)
                    that returns a single float or a TensorCollection of
                    floats, the comparison error. MSE by default.
            seed (int, optional): random seed
        """
        if error_func is None:
            def default_error_func(a, b):
                if getattr(a, 'device', None) != getattr(b, 'device', None):
                    if hasattr(a, 'cpu'):
                        a = a.cpu()
                    if hasattr(b, 'cpu'):
                        b = b.cpu()
                return cmap(batch_mse, a, b)

            error_func = default_error_func

        self.__dict__.update(config.__dict__)
        self.error_func = error_func
        self.random = np.random.RandomState(seed)

    def should_continue_warmup(self, step, x, F, z, h, transformer) -> bool:
        """Perform warm-up for `self.k_warmup` steps."""
        return step < self.k_warmup

    def should_end_comparison(self, step, x, F, z, h, transformer) -> bool:
        """Perform micro-macro comparison for `self.k_cmp` steps."""
        # For example, if k_cmp == 10, say that step #9 is the last one.
        return step >= self.k_cmp - 1

    def compute_error(self, x_macro, x_micro):
        error = self.error_func(x_macro, x_micro)
        if isinstance(error, TensorCollection):
            return sum(error.allvalues())
        else:
            return error

    def compute_and_check_error(self, uncertainty, x, F, z, h, transformer) \
            -> Tuple[Sequence[float], bool]:
        """Compute the real space micro-macro MSE and whether the error is
        small enough to use the macro solver.

        For debugging and analysis purposes, the error is computed always, even
        when uncertainty is too high to accept the macro solver.
        """
        with torch.no_grad():
            x_macro = transformer.inverse_transform(z)
            x_macro = cmap(lambda a, b: get_backend(a).cast_from(b),
                           x, x_macro)
        error = self.compute_error(x_macro, x)
        ok = uncertainty.max() < self.max_uncertainty \
                and error.max() < self.max_cmp_error
        return (error, ok)

    def should_accept_macro_step(self, step, uncertainty, F, z, h, transformer) -> bool:
        """Accept macro step if the uncertainty is below the uncertainty,
        limited to self.max_macro_steps."""
        return uncertainty.max() < self.max_uncertainty \
                and self._should_accept_step(step, self.max_macro_steps)

    def should_continue_micro(self, step, x, F) -> bool:
        return self._should_accept_step(step, self.max_micro_steps)

    def _should_accept_step(
            self,
            step: int,
            max_steps: Union[int, Tuple[int, int]]) -> bool:
        """Check if ``step`` is still below the given threshold ``max_steps``.

        ``max_steps`` can be one of the following:
            `int`: the maximum number of steps,
            `pair of ints`: range of maximum number of steps, inclusive.
        """
        if isinstance(max_steps, _int_like):
            return step < max_steps
        elif isinstance(max_steps, _tuple_like) and len(max_steps) == 2:
            if step < max_steps[0]:
                return True
            elif step < max_steps[1]:
                # This should produce a uniform distribution if invoked for
                # each `step` starting from 0...
                # return self.random.rand() > 1 / (max_steps[1] - step + 1)
                return self.random.rand() * (max_steps[1] - step + 1) > 1
            else:
                return False
        else:
            raise NotImplementedError(f"unrecognized type: {max_steps}")

    def should_continue_relaxation(self, step, x, F) -> bool:
        return step < self.num_relaxation_steps
