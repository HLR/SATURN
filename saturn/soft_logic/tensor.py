"""ProbabilisticTensor: the soft-logic engine (``&``, ``|``, ``~``, ``iota``, ``argmax`` over object variables).

Pure torch; must not import saturn.scene/perception/vlm/serving.
"""
from saturn.settings import env

import contextvars
import re

import numpy as np
import torch
import torch.nn.functional as F
from functools import wraps


def zero_superdiag_trailing(
    x: torch.Tensor, N: int | None = None, inplace: bool = False, fill: float = 0
) -> torch.Tensor:
    """
    Zero entries where the last N indices are all equal.
    If N is None, it is inferred as the count of trailing dims that all have the same size as the last dim.

    Examples
      x shape (1, 3, 3, 3) and N None  ⇒ N becomes 3
      x shape (B, d, d) and N 2         ⇒ zeros along the usual matrix diagonal for every batch item
    """
    if x.dim() == 0:
        return x

    # Infer N if not provided
    if N is None:
        d_last = x.size(-1)
        N = 1
        k = 2
        while k <= x.dim() and x.size(-k) == d_last:
            N += 1
            k += 1

    if N < 1 or N > x.dim():
        raise ValueError(f"N must be between 1 and {x.dim()}, got {N}.")

    # All last N dims must be the same size to define a super diagonal
    d = x.size(-1)
    for k in range(2, N + 1):
        if x.size(-k) != d:
            raise ValueError(
                "The last N dims must all be equal in size to zero the super diagonal."
            )

    i = torch.arange(d, device=x.device)
    idx = (slice(None),) * (x.dim() - N) + (i,) * N

    if inplace:
        x[idx] = fill
        return x
    else:
        y = x.clone()
        y[idx] = fill
        return y


def zero_equal_pairs(x: torch.Tensor, pairs, fill: float) -> torch.Tensor:
    """Set x[..., i, ..., i, ...] = fill for every (axis a, axis b) in `pairs`
    whenever the indices on axes a and b coincide. Axes are counted over the
    full tensor. Non-inplace; O(sum over pairs of N^(k-1)) writes, no k-dim mask."""
    y = x.clone()
    for a, b in pairs:
        d = y.shape[a]
        assert y.shape[b] == d, "zero_equal_pairs: axes must share a domain size"
        idx = torch.arange(d, device=y.device)
        index = [slice(None)] * y.dim()
        index[a] = idx
        index[b] = idx
        y[tuple(index)] = fill
    return y


def zero_any_equal_trailing(x: torch.Tensor, N: int, fill: float) -> torch.Tensor:
    """
    Zero entries where any pair among the last N indices are equal.
    Enforces all different over the last N dims.
    """
    d = x.size(-1)
    # Build a boolean mask over the last N dims
    mask = torch.zeros((d,) * N, dtype=torch.bool, device=x.device)
    for a in range(N):
        for b in range(a + 1, N):
            eye = torch.eye(d, dtype=torch.bool, device=x.device)
            view = [1] * N
            view[a] = d
            view[b] = d
            mask |= eye.view(*view)
    # Expand mask to the full tensor shape and apply
    full_mask = mask.expand(*x.shape[-N:])
    full_mask = full_mask.reshape((1,) * (x.dim() - N) + full_mask.shape)
    full_mask = full_mask.expand(*x.shape)
    return x.masked_fill(full_mask, fill)


# assign() on a joint tensor whose max is at or below this has no satisfying assignment.
_DEGENERATE_MAX = 1e-9


def check_satisfiable(tensor: torch.Tensor, vars) -> None:
    """Raise DegenerateBindingError if no joint assignment has non-zero truth.

    With ``&`` binding different variables to distinct entities, this happens
    when there are fewer entities than variables, or when one conjunct is
    zero everywhere. argmax would then silently return index 0 per variable.
    """
    # NaN is "no evidence" (see argmax); tensor.max() would return NaN and pass.
    if tensor.numel() == 0 or float(torch.nan_to_num(tensor, nan=0.0).max()) <= _DEGENERATE_MAX:
        from saturn.errors import DegenerateBindingError
        names = ", ".join(vars)
        raise DegenerateBindingError(
            f"assign(): the formula over ({names}) has no satisfying assignment "
            f"(every joint value is 0; shape {tuple(tensor.shape)}). Likely fewer "
            f"entities than variables (& binds distinct entities) or a predicate "
            f"that is 0 everywhere."
        )


# Variables range over OBJECTS; cameras enter a program only as constants
# (camera(n), pred[camera(n)]). Entity axes list the K objects
# first, then the cameras, so the executor sets K for the program it runs and
# assign() never binds a variable to a camera. Unset (None): every index binds.
# Temperature of the soft comparison between two formulas (training mode): a gap of
# 0.05 on the [0, 1] scale gives about 0.73.
_COMPARE_TAU = 0.05

BINDABLE_ENTITIES: contextvars.ContextVar = contextvars.ContextVar("BINDABLE_ENTITIES", default=None)

# Entity variables are x1, x2, ...; any other name (e.g. "o", the options of a
# question) ranges over its own domain: it is not an entity, so neither the
# object-only rule nor "distinct variables bind distinct entities" applies.
_ENTITY_VAR = re.compile(r"x\d+$")


def is_entity_var(v) -> bool:
    return bool(_ENTITY_VAR.match(str(v)))


def _objects_only(tensor: torch.Tensor, vars=None) -> torch.Tensor:
    k = BINDABLE_ENTITIES.get()
    if k is None or k <= 0:
        return tensor
    ent = [True] * tensor.dim() if vars is None else [is_entity_var(v) for v in vars]
    idx = tuple(slice(0, k) if (n > k and e) else slice(None) for n, e in zip(tensor.shape, ent))
    return tensor[idx]


def assign_indices(tensor: torch.Tensor, vars, mode: str = "max") -> dict:
    """{var: index} at the global max (or min) of *tensor*; NaN cells never win.
    Only object indices bind (see BINDABLE_ENTITIES)."""
    tensor = _objects_only(tensor, vars)
    if mode == "max":
        # The guard is about binding ENTITIES (fewer entities than variables,
        # a name that matches nothing). A formula over the options alone with
        # every value 0 means no option is supported: argmax keeps option
        # order, exactly as max(options, key=...) does.
        # A formula that is 0 everywhere (a named object matched nothing, or
        # fewer entities than variables) has no satisfying binding. Raise when
        # SAPY_STRICT_ASSIGN=1 (debugging); otherwise log it and keep the argmax
        # (index order), so the program still answers.
        if any(is_entity_var(v) for v in vars):
            try:
                check_satisfiable(tensor, vars)
            except Exception as e:
                if env("SAPY_STRICT_ASSIGN") == "1":
                    raise
                import logging
                logging.getLogger(__name__).warning(f"degenerate assign, falling back to argmax: {e}")
        flat_idx = torch.nan_to_num(tensor, nan=-float("inf")).argmax()
    elif mode == "min":
        flat_idx = torch.nan_to_num(tensor, nan=float("inf")).argmin()
    else:
        raise ValueError(f"mode must be 'max' or 'min', got '{mode}'.")
    coords = torch.unravel_index(flat_idx, tensor.shape)
    return {var: coords[i].item() for i, var in enumerate(vars)}


def _align_two(a, b):
    """Align two operands by variable name for an elementwise operation.

    Returns ``(ta, tb, vars)``: both tensors broadcast over the union of the
    operands' variables (order of first appearance), so ``f("x1") >= g("x2")``
    is a relation over (x1, x2) and ``f("x1") - f("x1")`` stays over x1. A plain
    number or an unnamed tensor is a constant over the other operand's variables.
    """
    def unpack(o):
        if isinstance(o, ProbabilisticTensor):
            return o.tensor, list(o.vars)
        if hasattr(o, "to_tensor") and not isinstance(o, torch.Tensor):
            o = o.to_tensor()
            return o.tensor, list(o.vars)
        return (o if isinstance(o, torch.Tensor) else torch.as_tensor(o)), None
    ta, va = unpack(a)
    tb, vb = unpack(b)
    if va is None or vb is None or va == vb or not va or not vb:
        vars_ = va if va is not None and (va or vb is None) else (vb or [])
        tb = tb.to(ta.device) if isinstance(tb, torch.Tensor) else tb
        return ta, tb, list(vars_)
    out_vars = list(va) + [v for v in vb if v not in va]
    sizes = {}
    for t, vs in ((ta, va), (tb, vb)):
        for i, v in enumerate(vs):
            n = int(t.shape[i])
            if v in sizes and sizes[v] != n:
                raise ValueError(f"Inconsistent domain size for variable {v}.")
            sizes[v] = n
    shape = [sizes[v] for v in out_vars]

    def expand(t, vs):
        t = t.permute([vs.index(v) for v in out_vars if v in vs])
        for i, v in enumerate(out_vars):
            if v not in vs:
                t = t.unsqueeze(i)
        return t.expand(*shape)
    return expand(ta, va), expand(tb.to(ta.device), vb), out_vars


def invert_tensor(t: torch.Tensor, vars=None) -> torch.Tensor:
    """NOT: 1 - t, with cells where two same-size axes share an index set to 0.

    Distinct variables bind distinct entities, so a self-pair (x1 == x2) is
    never true under NOT either.
    """
    out = 1 - t
    pairs = [(a, b) for a in range(t.dim()) for b in range(a + 1, t.dim())
             if t.shape[a] == t.shape[b]
             and (vars is None or (is_entity_var(vars[a]) and is_entity_var(vars[b])))]
    return zero_equal_pairs(out, pairs, fill=0.0) if pairs else out


def _to_common_device(concepts):
    """Bring operands onto one device without mutating the callers' tensors."""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return [c if c.tensor.device == device else _rebind(c, c.tensor.to(device)) for c in concepts]


def _rebind(concept, tensor):
    """Shallow copy of a ProbabilisticTensor with a different underlying tensor."""
    new = ProbabilisticTensor.__new__(ProbabilisticTensor)
    new.__dict__.update(concept.__dict__)
    new.tensor = tensor
    return new



# ---------------------------------------------------------------------------
# Constants in formulas: numbers are truth degrees, entity indices bind a slot.
# ---------------------------------------------------------------------------

def is_truth_constant(x) -> bool:
    """A plain number (or bool) used as a truth degree inside a formula."""
    return isinstance(x, (bool, int, float, np.number, np.bool_)) and not isinstance(x, torch.Tensor)


def _lift_constants(concepts):
    """Numbers and bools in ``&`` / ``|`` are constant truths: a 0-d formula
    with no variables (``J & (yaw > 0)`` is J where the claim holds, else 0)."""
    return [ProbabilisticTensor(torch.tensor(float(c), dtype=torch.float64), vars=[])
            if is_truth_constant(c) else c for c in concepts]


def is_entity_index(a) -> bool:
    """An entity index: an int (object index or camera(N)), never a bool."""
    if isinstance(a, (bool, np.bool_)):
        return False
    if isinstance(a, (int, np.integer)):
        return True
    return isinstance(a, torch.Tensor) and a.dim() == 0 and not a.is_floating_point()


def bind_arguments(pt, args):
    """``pred("x1", camera(2))``: each argument is a variable name (the slot stays
    free) or an entity index (the slot is fixed to that entity). With every slot
    fixed the result is a formula with no variables: its value is the predicate's
    truth for those entities, so it still composes with & | ~ and .exists()."""
    ndim = pt.tensor.dim()
    if len(args) != ndim:
        raise TypeError(
            f"this predicate relates {ndim} entit{'y' if ndim == 1 else 'ies'}; called with "
            f"{len(args)} argument(s) {list(args)!r}. Pass one variable name (\"x1\") or entity "
            f"index (an object index or camera(N)) per entity.")
    key, names = [], []
    for pos, a in enumerate(args):
        if isinstance(a, str):
            key.append(slice(None))
            names.append(a)
        elif is_entity_index(a):
            i, n = int(a), int(pt.tensor.shape[pos])
            if not 0 <= i < n:
                raise IndexError(f"entity index {i} is out of range: this predicate covers entities 0..{n - 1}")
            key.append(i)
        else:
            kind = type(a).__name__
            shape = getattr(a, "shape", None)
            hint = (" A 3D point is not an entity: anchor.first_person.<dir>[point] scores a point "
                    "relative to the anchor." if shape is not None and tuple(shape) == (3,) else "")
            raise TypeError(
                f"a predicate argument is a variable name (\"x1\") or an entity index (an object "
                f"index or camera(N)); got {kind} {a!r}.{hint}")
    sub = pt[tuple(key)]
    return sub(*names) if names else sub


def _align_concepts(concepts, op_name: str):
    """Bring the operands of an n-ary connective onto one variable layout.

    Each operand is a ProbabilisticTensor (``.tensor`` with one axis per name in
    ``.vars``), a bare torch tensor (its axes take the leading variables of the
    others), a PredicateArray or a constant. Variables are ordered by first
    appearance; each keeps its own domain size (objects, cameras, regions). Every
    operand is permuted, unsqueezed and broadcast to that global shape.

    Returns ``(global_vars, var_sizes, global_shape, expanded_tensors)``.
    """
    # PredicateArray (numpy score vector) lifts itself into the algebra.
    concepts = [c.to_tensor() if hasattr(c, "to_tensor") and not isinstance(c, torch.Tensor) else c
                for c in concepts]
    concepts = _lift_constants(concepts)
    if not concepts:
        raise ValueError(f"No concepts provided for {op_name} operation.")

    # Global order of variables: order of first appearance.
    global_vars = []
    for concept in concepts:
        if isinstance(concept, torch.Tensor):
            continue
        for var in concept.vars:
            if var not in global_vars:
                global_vars.append(var)
    # A bare tensor takes the leading global variables, one per axis.
    for i, concept in enumerate(concepts):
        if isinstance(concept, torch.Tensor):
            concepts[i] = ProbabilisticTensor(concept, vars=global_vars[: len(concept.shape)])
    # Bring operands onto one device without mutating the callers' tensors.
    concepts = _to_common_device(concepts)

    # Domain size of each variable; every operand that uses a variable must agree on it.
    var_sizes = {}
    for concept in concepts:
        for i, var in enumerate(concept.vars):
            size = 0 if len(concept.tensor.shape) == 0 else concept.tensor.shape[i]
            if var in var_sizes and var_sizes[var] != size:
                raise ValueError(f"Inconsistent domain size for variable {var}.")
            var_sizes[var] = size
    global_shape = [var_sizes[var] for var in global_vars]
    target_numel = 1
    for size in global_shape:
        target_numel *= size

    expanded_tensors = []
    for concept in concepts:
        local_vars = concept.vars
        # (a) Permute the axes into global order.
        perm_order = [local_vars.index(var) for var in global_vars if var in local_vars]
        new_tensor = concept.tensor
        if perm_order != list(range(len(local_vars))):
            new_tensor = new_tensor.permute(perm_order)
        # (b) Insert a size-1 axis for every global variable the operand lacks.
        for i, var in enumerate(global_vars):
            if var not in local_vars:
                new_tensor = new_tensor.unsqueeze(i)
        # (c) Broadcast to the global shape.
        if global_shape:
            if new_tensor.numel() == target_numel:
                new_tensor = new_tensor.reshape(global_shape)
            elif new_tensor.dim() > len(global_shape):
                new_tensor = new_tensor.squeeze(0)
            new_tensor = new_tensor.expand(*global_shape)
        expanded_tensors.append(new_tensor)
    return global_vars, var_sizes, global_shape, expanded_tensors


def and_op(*concepts):
    """AND of any number of probabilistic concepts: the element-wise minimum after
    aligning their variables (see ``_align_concepts``). Variables over the same
    entity domain bind distinct entities (``ProbabilisticTensor.forbid_equal``)."""
    global_vars, var_sizes, global_shape, expanded_tensors = _align_concepts(concepts, "AND")
    # Fold pairwise (min = probabilistic AND). A stack+amin materialises
    # k x N^k at once; the fold peaks at 2 x N^k.
    result_tensor = expanded_tensors[0]
    for t in expanded_tensors[1:]:
        result_tensor = torch.minimum(result_tensor, t)
    if len(expanded_tensors) == 1:
        result_tensor = result_tensor.clone()   # never alias an operand
    N = len(global_vars)
    # "Distinct variables bind distinct entities" only makes sense between
    # variables that range over the SAME domain (same size). An object variable
    # and a camera variable that happen to share an index are not "equal".
    same_domain_pairs = [
        (a, b) for a in range(N) for b in range(a + 1, N)
        if var_sizes[global_vars[a]] == var_sizes[global_vars[b]]
        and is_entity_var(global_vars[a]) and is_entity_var(global_vars[b])
    ]
    if same_domain_pairs and N >= 2:
        if ProbabilisticTensor.forbid_equal == "super" and len(set(global_shape)) == 1:
            result_tensor = zero_superdiag_trailing(result_tensor, N, inplace=False, fill=0.0)
        elif ProbabilisticTensor.forbid_equal in ("super", "any"):
            result_tensor = zero_equal_pairs(result_tensor, same_domain_pairs, fill=0.0)

    return ProbabilisticTensor(result_tensor, vars=global_vars)


def or_op(*concepts):
    """OR of any number of probabilistic concepts: the element-wise maximum after
    aligning their variables (see ``_align_concepts``)."""
    global_vars, _, _, expanded_tensors = _align_concepts(concepts, "OR")
    result_tensor = torch.stack(expanded_tensors, dim=0).amax(dim=0)
    return ProbabilisticTensor(result_tensor, vars=global_vars)


def serializable(data):
    if data is None:
        return None
    if data is True or data is False:
        return data
    if isinstance(data, (int, float, str)):
        return data
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy().tolist()
    if isinstance(data, list):
        return [serializable(d) for d in data]
    if isinstance(data, dict):
        return {k: serializable(v) for k, v in data.items()}
    if isinstance(data, tuple):
        return tuple(serializable(d) for d in data)
    if isinstance(data, set):
        return {serializable(d) for d in data}
    if isinstance(data, ProbabilisticTensor):
        return data.tensor.detach().cpu().to(torch.float32).numpy().tolist()
    if data is Ellipsis:
        return "..."
    if isinstance(data, slice):
        return {
            "start": getattr(data, "start", None),
            "stop": getattr(data, "stop", None),
            "step": getattr(data, "step", None),
            "indices": getattr(data, "indices", None),
        }
    elif callable(
        data
    ):  # Handles cases where the indices method is mistakenly included
        return str(data)

    return data


class ProbabilisticTensor:
    training = False
    # Execution trace. A ContextVar gives each asyncio task (and each thread
    # started via asyncio.to_thread, which copies the context) its own trace,
    # so concurrently executing samples never share one.
    _cache_var: "contextvars.ContextVar" = contextvars.ContextVar(
        "probabilistic_tensor_cache", default=None
    )

    forbid_equal: str = "any"

    @classmethod
    def start_cache(cls):
        """Begin a trace scoped to the calling task/thread."""
        cls._cache_var.set([])

    @classmethod
    def end_cache(cls):
        """Return this task's trace and clear it. Never sees other tasks' entries."""
        trace = cls._cache_var.get()
        cls._cache_var.set(None)
        return list(trace) if trace is not None else []


    def __float__(self):
        # Allow generated programs to call float() on a scalar PT (e.g. score(...)[idx])
        t = self.tensor
        if t.numel() == 1:
            return float(t.item())
        raise TypeError(
            f"Cannot convert multi-element ProbabilisticTensor (shape {tuple(t.shape)}) to float"
        )

    def __init__(self, tensor, vars=None, normalized=False, extra_info=None):
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor)
        self.tensor = tensor
        self.normalized = normalized
        self.vars = (
            vars
            if vars is not None
            else [f"x{i}" for i in range(1, len(tensor.shape) + 1)]
        )
        if extra_info is not None:
            inputs = {
                "args": (extra_info),
                "kwargs": {},
                "self.tensor": serializable(self),
            }
            self.__class__._cache_action("__init__", inputs, None)

    def __call__(self, *args):
        if any(not isinstance(a, str) for a in args):
            return bind_arguments(self, args)   # constants: pred("x1", camera(2))
        return ProbabilisticTensor(self.tensor, vars=args)

    @classmethod
    def _cache_action(cls, action_name, inputs, result):
        """Helper method to cache an action."""
        trace = cls._cache_var.get()
        if trace is None:
            # Tracing not started for this context: drop the record rather than
            # leaking it into an unrelated sample's trace.
            return
        trace.append(
            {
                "action": action_name,
                "inputs": serializable(inputs),
                "result": serializable(result),
            }
        )

    @staticmethod
    def cache_action(method):
        """Decorator to cache method calls."""

        @wraps(method)
        def wrapper(self, *args, **kwargs):
            result = method(self, *args, **kwargs)
            # Serialising calls `.tolist()` on the WHOLE tensor, which is slow
            # for large joints and holds the GIL. Only pay it when a trace is
            # actually being recorded.
            if self.__class__._cache_var.get() is None:
                return result
            action_name = method.__name__
            # Prepare inputs for caching
            inputs = {
                "args": [serializable(arg) for arg in args],
                "kwargs": serializable(kwargs),
                "self.tensor": serializable(self),
            }
            self.__class__._cache_action(action_name, inputs, result)
            return result

        return wrapper

    def __repr__(self):
        return str(self.tensor)

    def __str__(self):
        return str(self.tensor)

    @cache_action.__get__(object)
    def __and__(self, other):
        return and_op(self, other)


    @cache_action.__get__(object)
    def __rand__(self, other):
        return and_op(other, self)

    def __ror__(self, other):
        return or_op(other, self)

    ## OR

    @cache_action.__get__(object)
    def __or__(self, other):
        return or_op(self, other)

    ## not
    @cache_action.__get__(object)
    def __invert__(self):
        return ProbabilisticTensor(invert_tensor(self.tensor, self.vars), vars=self.vars)

    @cache_action.__get__(object)
    def __gt__(self, other):
        # Against a constant (a threshold test such as score > 0.5) the training
        # form keeps its margin. Between two formulas it is a symmetric soft
        # comparison: gt(a, b) + gt(b, a) = 1 (and 0.5 at a tie).
        ta, tb, vars_ = _align_two(self, other)
        if self.training:
            if isinstance(other, ProbabilisticTensor):
                rv = torch.sigmoid((ta - tb) / _COMPARE_TAU)
            else:
                rv = torch.sigmoid((ta - tb + 0.25) / 0.25)
        else:
            rv = (ta > tb).float()
        return ProbabilisticTensor(rv, vars=vars_)
    @cache_action.__get__(object)
    def __lt__(self, other):
        ta, tb, vars_ = _align_two(self, other)
        if self.training:
            if isinstance(other, ProbabilisticTensor):
                rv = torch.sigmoid((tb - ta) / _COMPARE_TAU)
            else:
                rv = torch.sigmoid((tb - ta + 0.25) / 0.25)
        else:
            rv = (ta < tb).float()
        return ProbabilisticTensor(rv, vars=vars_)
    @cache_action.__get__(object)
    def __eq__(self, other):
        _count_margin = 0.25
        _count_tau = 0.25
        ta, tb, vars_ = _align_two(self, other)
        rv = torch.sigmoid(
            (2 * _count_margin - (ta - tb).abs())
            / (2 * _count_margin)
            / _count_tau
        )
        return ProbabilisticTensor(rv, vars=vars_)
    @cache_action.__get__(object)
    def __ne__(self, other):
        eq = self.__eq__(other)
        return ProbabilisticTensor(1.0 - eq.tensor, vars=eq.vars)
    __hash__ = object.__hash__   # defining __eq__ would otherwise make instances unhashable

    # Outside training, >= and <= are crisp like > and <: the soft __eq__ is
    # ~0.92 at a 0.2 gap, so gt|eq would make PT(0.3) >= 0.5 true.
    @cache_action.__get__(object)
    def __ge__(self, other):
        if not self.training:
            ta, tb, vars_ = _align_two(self, other)
            return ProbabilisticTensor((ta >= tb).float(), vars=vars_)
        if isinstance(other, ProbabilisticTensor):
            return self.__gt__(other)   # the symmetric soft comparison is already non-strict at a tie
        return self.__gt__(other) | self.__eq__(other)
    @cache_action.__get__(object)
    def __le__(self, other):
        if not self.training:
            ta, tb, vars_ = _align_two(self, other)
            return ProbabilisticTensor((ta <= tb).float(), vars=vars_)
        if isinstance(other, ProbabilisticTensor):
            return self.__lt__(other)
        return self.__lt__(other) | self.__eq__(other)
    @cache_action.__get__(object)
    def __add__(self, other):
        ta, tb, vars_ = _align_two(self, other)
        return ProbabilisticTensor(torch.add(ta, tb), vars=vars_)
    @cache_action.__get__(object)
    def __mul__(self, other):
        ta, tb, vars_ = _align_two(self, other)
        return ProbabilisticTensor(torch.mul(ta, tb), vars=vars_)
    @cache_action.__get__(object)
    def __truediv__(self, other):
        ta, tb, vars_ = _align_two(self, other)
        return ProbabilisticTensor(torch.div(ta, tb), vars=vars_)
    @cache_action.__get__(object)
    def __floordiv__(self, other):
        ta, tb, vars_ = _align_two(self, other)
        return ProbabilisticTensor(torch.floor_divide(ta, tb), vars=vars_)
    @cache_action.__get__(object)
    def __mod__(self, other):
        ta, tb, vars_ = _align_two(self, other)
        return ProbabilisticTensor(torch.remainder(ta, tb), vars=vars_)
    @cache_action.__get__(object)
    def __pow__(self, other):
        ta, tb, vars_ = _align_two(self, other)
        return ProbabilisticTensor(torch.pow(ta, tb), vars=vars_)
    @cache_action.__get__(object)
    def __abs__(self):
        return ProbabilisticTensor(self.tensor.abs(), vars=self.vars)

    @cache_action.__get__(object)
    def __neg__(self):
        return ProbabilisticTensor(-self.tensor, vars=self.vars)

    @cache_action.__get__(object)
    def __pos__(self):
        return self

    @cache_action.__get__(object)
    def exists(self, *vars, **kwargs):
        # A single variable may also be passed as the 'var' keyword argument.
        if "var" in kwargs:
            if vars:
                raise ValueError(
                    "Cannot specify both positional arguments and 'var' keyword argument."
                )
            if kwargs["var"] is not None:
                vars = (kwargs["var"],)

        # If vars is empty (no positional args and var=None or not provided), perform scalar existence
        if not vars:
            # A scalar binds no variable (a placeholder var would give and_op a size-0 axis).
            if self.tensor.numel() == 0:
                return ProbabilisticTensor(torch.tensor(0.0), vars=[])
            t = _objects_only(self.tensor, self.vars)      # only objects bind an entity variable
            if t.numel() == 0:
                return ProbabilisticTensor(torch.tensor(0.0), vars=[])
            return ProbabilisticTensor(t.amax(), vars=[])

        # Projected existence over multiple variables
        dims_to_reduce = []
        for v in vars:
            if v in self.vars:
                dims_to_reduce.append(self.vars.index(v))
            else:
                raise ValueError(
                    f"Variable {v} not found in tensor variables {self.vars}"
                )

        # Remove duplicate dimensions.
        dims_to_reduce = tuple(set(dims_to_reduce))

        # Only objects bind the quantified entity variables (the kept axes keep their size).
        t = self.tensor
        k = BINDABLE_ENTITIES.get()
        if k is not None and k > 0:
            idx = tuple(slice(0, k) if (d in dims_to_reduce and t.shape[d] > k and is_entity_var(self.vars[d]))
                        else slice(None) for d in range(t.dim()))
            t = t[idx]
        reduced_tensor = t.amax(dim=dims_to_reduce)

        new_vars = [v for i, v in enumerate(self.vars) if i not in dims_to_reduce]
        return ProbabilisticTensor(reduced_tensor, vars=new_vars)

    @cache_action.__get__(object)
    def count(self, *vars):
        """How many entities satisfy the formula.

        Without arguments it counts values of the formula's first entity variable
        (other variables are projected out with exists first), so one chair that
        satisfies the formula with three anchors counts once. With variable names
        it keeps those axes and counts over the rest. At inference an entity counts
        when its degree is at least 0.5 (the 0.5-cut); in training the count is the
        sigma-count, the sum of degrees. Only objects bind an entity variable.
        """
        t = self.tensor
        vars_ = list(self.vars)
        if not vars and vars_:
            keep = next((v for v in vars_ if is_entity_var(v)), vars_[0])
            drop = tuple(i for i, v in enumerate(vars_) if v != keep)
            if drop:
                t = t.amax(dim=drop)
            vars_ = [keep]
            t = _objects_only(t, vars_)
            counted = t if self.training else (t >= 0.5).float()
            return counted.sum()
        counted = t if self.training else (t >= 0.5).float()
        if not vars:
            return counted.sum()
        drop = tuple(i for i, v in enumerate(vars_) if v not in vars)
        if drop:
            counted = counted.sum(dim=drop)
        return ProbabilisticTensor(counted, vars=[v for v in vars_ if v in vars])
    @cache_action.__get__(object)
    def argmax(self, var=None, k=None, dim=None, object_id=None):
        # Guard: empty tensor → return -1 sentinel (no objects)
        if self.tensor.numel() == 0:
            return -1
        if torch.isnan(self.tensor).any():
            # NaN is "no evidence", never a winner (torch.argmax returns NaN's index)
            return _rebind(self, torch.nan_to_num(self.tensor, nan=-float("inf"))).argmax(
                var=var, k=k, dim=dim, object_id=object_id
            )

        if var is not None:
            if isinstance(var, str):
                t = _objects_only(self.tensor, self.vars)    # only objects bind an entity variable
                corrds = torch.unravel_index(t.argmax(), t.shape)
                var_index = self.vars.index(var)
                return corrds[var_index].item()

            else:
                raise ValueError("var must be a valid variable.")
        if object_id is not None:
            # If an object ID is provided, find the argmax for that specific object
            return int(
                self.tensor[object_id].argmax(dim=dim).detach().cpu().numpy().item()
            )

        if k is not None:
            # If k is provided, use topk to get the top k elements
            return int(self.tensor.topk(k)[1][k - 1])  # [1] returns the indices

        # Default behavior, return the argmax of the entire tensor along the specified dimension
        if len(self.tensor.shape) > 1 and self.tensor.shape[1] == self.tensor.shape[0]:
            ## first sum the tensor along the row
            return int(
                self.tensor.sum(dim=1).argmax(dim=dim).detach().cpu().numpy().item()
            )
        return int(self.tensor.argmax(dim=dim).detach().cpu().numpy().item())

    @cache_action.__get__(object)
    def assign(self, mode="max"):
        """
        Returns a dict mapping each variable name to the index along that dimension
        at the global max (or min) position of the tensor.

        For a 1D tensor with vars ["x1"]:
            assign("max") -> {"x1": 5}  (index of the max element)
        For a 2D tensor with vars ["x1", "x2"]:
            assign("max") -> {"x1": 2, "x2": 4}  (indices such that tensor[2, 4] is max)

        Args:
            mode (str): "max" (default) or "min".

        Returns:
            dict: {var_name: index, ...}

        Raises:
            DegenerateBindingError: mode "max" on a tensor with no non-zero entry.
        """
        return assign_indices(self.tensor, self.vars, mode)

    @cache_action.__get__(object)
    def topk(self, var=None, k=1, dim=-1):
        """
        Returns the top k indices along the specified dimension.
        """
        if var is not None:
            var_index = self.vars.index(var)
            return self.tensor.topk(k=k, dim=var_index)[1]
        else:
            return self.tensor.topk(k=k, dim=dim)[1]

    @cache_action.__get__(object)
    def argmin(self, var=None, k=None, dim=None, object_id=None):
        # Mirrors argmax: empty -> -1, NaN never wins, default returns an int.
        if self.tensor.numel() == 0:
            return -1
        if torch.isnan(self.tensor).any():
            return _rebind(self, torch.nan_to_num(self.tensor, nan=float("inf"))).argmin(
                var=var, k=k, dim=dim, object_id=object_id
            )
        if var is not None:
            if isinstance(var, str):
                corrds = torch.unravel_index(self.tensor.argmin(), self.tensor.shape)
                var_index = self.vars.index(var)
                return corrds[var_index].item()

            else:
                raise ValueError("var must be a valid variable.")
        if object_id is not None:
            # If an object ID is provided, find the argmin for that specific object
            return int(
                self.tensor[object_id].argmin(dim=dim).detach().cpu().numpy().item()
            )

        if k is not None:
            # If k is provided, use topk to get the top k elements
            return int(self.tensor.topk(k, largest=False)[1][k - 1])

        if len(self.tensor.shape) > 1 and self.tensor.shape[1] == self.tensor.shape[0]:
            return int(
                self.tensor.sum(dim=1).argmin(dim=dim).detach().cpu().numpy().item()
            )
        return int(self.tensor.argmin(dim=dim).detach().cpu().numpy().item())

    @cache_action.__get__(object)
    def argsort(self, var=None, dim=-1):
        """
        Returns the indices that would sort the tensor along the specified dimension.
        """
        if var is not None:
            var_index = self.vars.index(var)
            return self.tensor.argsort(dim=var_index)
        else:
            return self.tensor.argsort(dim=dim)

    @cache_action.__get__(object)
    def max(self, var=None, dim=-1):
        return ProbabilisticTensor(self.tensor.max(dim=dim).values)

    def amax(self, dim=-1):
        return ProbabilisticTensor(self.tensor.amax(dim=dim))

    @cache_action.__get__(object)
    def min(self, dim=-1):
        return ProbabilisticTensor(self.tensor.min(dim=dim).values)

    @cache_action.__get__(object)
    def amin(self, dim=-1):
        return ProbabilisticTensor(self.tensor.amin(dim=dim))

    @cache_action.__get__(object)
    def sum(self, dim=-1):
        return ProbabilisticTensor(self.tensor.sum(dim=dim))

    @cache_action.__get__(object)
    def __bool__(self):
        return (self.tensor >= 0.5).all().item()

    @cache_action.__get__(object)
    def __len__(self):
        return len(self.tensor)

    @cache_action.__get__(object)
    def __getitem__(self, key):
        if isinstance(key, str) and key in self.vars:
            key = self.vars.index(key)
        elif isinstance(key, tuple):
            key = tuple(
                [
                    int(k.cpu().numpy().item())
                    if isinstance(k, torch.Tensor) and k.dim() == 0
                    else k
                    for k in key
                ]
            )
        elif isinstance(key, torch.Tensor) and key.dim() == 0:
            key = int(key.cpu().numpy().item())
        new_vars = self._vars_after_indexing(key)
        return ProbabilisticTensor(self.tensor.__getitem__(key), vars=new_vars)

    def _vars_after_indexing(self, key):
        """Compute the var list for the tensor returned by self.tensor[key].

        Rules (per key position, aligned with tensor axes):
          int / 0-d tensor / np scalar      → axis collapsed, var dropped
          slice / list / 1-d+ tensor / mask → axis kept, var preserved
          None / np.newaxis                 → new axis inserted (placeholder)
          Ellipsis                          → expands to enough slice(None)s
        Untouched trailing axes are kept implicitly.
        """
        if not self.vars:
            return list(self.vars) if self.vars else []

        items = key if isinstance(key, tuple) else (key,)

        if any(k is Ellipsis for k in items):
            n_explicit = sum(
                1 for k in items if k is not Ellipsis and k is not None
            )
            n_fill = max(0, self.tensor.ndim - n_explicit)
            expanded = []
            seen_ellipsis = False
            for k in items:
                if k is Ellipsis:
                    if not seen_ellipsis:
                        expanded.extend([slice(None)] * n_fill)
                        seen_ellipsis = True
                else:
                    expanded.append(k)
            items = tuple(expanded)

        new_vars = []
        axis = 0
        for k in items:
            if k is None:
                new_vars.append("_newaxis")
                continue
            collapses = (
                (isinstance(k, int) and not isinstance(k, bool))
                or (isinstance(k, torch.Tensor) and k.dim() == 0)
                or (
                    hasattr(k, "ndim")
                    and hasattr(k, "__index__")
                    and getattr(k, "ndim", None) == 0
                )
            )
            if collapses:
                axis += 1
                continue
            if axis < len(self.vars):
                new_vars.append(self.vars[axis])
            axis += 1

        while axis < len(self.vars):
            new_vars.append(self.vars[axis])
            axis += 1

        return new_vars

    @cache_action.__get__(object)
    def __setitem__(self, key, value):
        self.tensor[key] = value.tensor

    @cache_action.__get__(object)
    def normalize(self, dim=-1):
        ## perform linear normalization with zero as min
        lo = self.tensor.min(dim=dim, keepdim=True).values
        hi = self.tensor.max(dim=dim, keepdim=True).values
        return ProbabilisticTensor((self.tensor - lo) / (hi - lo + 1e-8), vars=self.vars)

    @cache_action.__get__(object)
    def __sub__(self, other):
        ta, tb, vars_ = _align_two(self, other)
        return ProbabilisticTensor(torch.sub(ta, tb), vars=vars_)
    @cache_action.__get__(object)
    def __rsub__(self, other):
        tb, ta, vars_ = _align_two(self, other)
        return ProbabilisticTensor(torch.sub(ta, tb), vars=vars_)
    @cache_action.__get__(object)
    def any(self, **kwargs):
        ## support dim and axis or -1
        dim = kwargs.get("dim", kwargs.get("axis", -1))
        return ProbabilisticTensor(self.tensor.any(dim=dim))

    @cache_action.__get__(object)
    def all(self, dim=-1):
        tensor = self.tensor
        if len(self.tensor.shape) > 1 and self.tensor.shape[1] == self.tensor.shape[0]:
            tensor = tensor + torch.eye(tensor.shape[0])
        return (self.tensor > 0.5).all(dim=dim)

    @cache_action.__get__(object)
    def forall(self, var=None, dim=-1):
        """Fuzzy universal quantifier: the minimum over ``var`` (inf over a finite domain).

        Cells where ``var`` would bind the same entity as another entity variable
        of the same domain are excluded (distinct variables bind distinct entities),
        and only objects bind an entity variable (see BINDABLE_ENTITIES). An empty
        domain gives 1. Without ``var`` the minimum is over all cells (or ``dim``
        when given explicitly for an unnamed tensor).
        """
        t = self.tensor
        if var is None:
            if not self.vars or dim != -1:
                return ProbabilisticTensor(t.amin(dim=dim) if t.dim() else t, vars=[])
            var_list = list(self.vars)
        else:
            if var not in self.vars:
                raise ValueError(f"Variable {var} not found in tensor variables {self.vars}")
            var_list = [var]
        t = t.clone()
        for v in var_list:
            if not is_entity_var(v):
                continue
            a = self.vars.index(v)
            k = BINDABLE_ENTITIES.get()
            if k is not None and 0 < k < t.shape[a]:
                idx = [slice(None)] * t.dim(); idx[a] = slice(k, None)
                t[tuple(idx)] = 1.0               # non-object slots never falsify
            for b, w in enumerate(self.vars):
                if b != a and is_entity_var(w) and t.shape[b] == t.shape[a]:
                    n = t.shape[a]
                    eye = torch.eye(n, dtype=torch.bool, device=t.device)
                    shape = [1] * t.dim(); shape[a] = n; shape[b] = n
                    t = torch.where(eye.reshape(shape).expand_as(t), torch.ones_like(t), t)
        dims = tuple(self.vars.index(v) for v in var_list)
        if t.numel() == 0:
            return ProbabilisticTensor(torch.tensor(1.0), vars=[v for v in self.vars if v not in var_list])
        return ProbabilisticTensor(t.amin(dim=dims), vars=[v for v in self.vars if v not in var_list])
    @cache_action.__get__(object)
    def mask(self, other):
        """
        Mask the current tensor with another tensor.
        """
        if self.tensor.shape == other.tensor.shape:
            return ProbabilisticTensor(
                torch.stack([self.tensor, other.tensor], dim=-1).prod(dim=-1),
                vars=self.vars,
            )
        else:
            if len(self.tensor.shape) == 1:
                a = self.normalize().tensor
                b = other.tensor
            else:
                a = other.normalize().tensor
                b = self.tensor
            for _ in range(len(b.shape)):
                b = a.unsqueeze(1).repeat((1, a.shape[0])) * b
                b = b.transpose(0, 1)
            return ProbabilisticTensor(b, vars=self.vars)

    @cache_action.__get__(object)
    def implies(self, other, logic="kleene"):
        if logic == "lukasiewicz":
            ta, tb, vars_ = _align_two(self, other)
            return ProbabilisticTensor(torch.clamp(1 - ta + tb, max=1.0), vars=vars_)
        if logic != "kleene":
            raise ValueError(f"implies: logic must be 'kleene' or 'lukasiewicz', got {logic!r}")
        return ~self | other

    @cache_action.__get__(object)
    def not_implies(self, other):
        """
        Negated Implication: A & ~B
        """
        return self & ~other

    @cache_action.__get__(object)
    def __xor__(self, other):
        """
        Logical XOR: (A | B) & ~(A & B)
        """
        return (self | other) & ~(self & other)

    @cache_action.__get__(object)
    def __rshift__(self, other):
        """
        Logical IMPLIES: A >> B ≡ ~A | B
        """
        return ~self | other

    @cache_action.__get__(object)
    def __lshift__(self, other):
        """
        NOT IMPLIES: A << B ≡ A & ~B
        """
        return self & ~other

    @cache_action.__get__(object)
    def iota(self, var=None, method="linear"):
        """
        IOTA operator: Aggregates over other variables and normalizes the result
        to create a distribution or selection score for the target variable.

        Args:
            var (str): The variable to create a distribution for. Defaults to the first variable.
            method (str): 'softmax' or 'linear' normalization.

        Returns:
            ProbabilisticTensor: Normalized 1D tensor representing the distribution
                                 over the specified variable. The `vars` will contain only `var`.
        """
        if method not in ("softmax", "linear"):
            raise ValueError(f"Unknown normalization method: {method}. Use 'softmax' or 'linear'.")
        if not self.vars:
            if self.tensor.numel() == 1:
                # A scalar has nothing to normalize against: softmax gives 1.0,
                # linear returns it unchanged.
                if method == "softmax":
                    return ProbabilisticTensor(
                        self.tensor / self.tensor, vars=self.vars, normalized=True
                    )
                else:  # Linear normalization of scalar is ambiguous, return original
                    return self
            else:
                raise ValueError(
                    "Cannot perform iota on tensor with dimensions but no variable names."
                )

        if var is None:
            var = self.vars[0]  # Default to the first variable

        if var not in self.vars:
            # If the tensor is 1D with a single variable, allow iota with a
            # different variable name — the result will be relabeled.  This
            # handles a common LLM code-gen pattern: score(...).iota("x2").
            if self.tensor.dim() == 1 and len(self.vars) == 1:
                result = self.iota(self.vars[0])
                result.vars = [var]
                return result
            raise ValueError(f"Variable '{var}' not in tensor vars: {self.vars}")

        var_index = self.vars.index(var)
        # NaN = no evidence, never a winner; copy so in-place steps cannot leak out
        src = torch.nan_to_num(self.tensor, nan=0.0)
        num_dims = self.tensor.dim()

        # --- Aggregation Step ---
        dims_to_aggregate = [i for i in range(num_dims) if i != var_index]

        if dims_to_aggregate:
            # Aggregate using maximum possibility across other dimensions
            aggregated_tensor = src.amax(dim=tuple(dims_to_aggregate))
        else:
            # Tensor is already 1D (or potentially 0D if original was 0D).
            aggregated_tensor = src

        # Ensure tensor is at least 1D for normalization if it became scalar after aggregation
        if aggregated_tensor.dim() == 0:
            aggregated_tensor = aggregated_tensor.unsqueeze(0)  # Make it 1D

        # --- Normalization Step ---
        if aggregated_tensor.numel() == 0:
            # Handle empty tensor case
            normalized_tensor = torch.empty(
                0, dtype=aggregated_tensor.dtype, device=aggregated_tensor.device
            )
        elif method == "softmax":
            # Softmax requires at least 1D, which aggregated_tensor should be now.
            normalized_tensor = F.softmax(
                aggregated_tensor, dim=0
            )  # Normalize the single dimension
        else:
            # Linear normalization (scale to [0, 1])
            min_val = aggregated_tensor.min()
            max_val = aggregated_tensor.max()

            aggregated_tensor = aggregated_tensor + 1e-6
            range_val = max_val - min_val
            if range_val.item() < 1e-6:
                # All values are (nearly) equal: uniform 1/N if max_val > 0, else zeros.
                num_elements = aggregated_tensor.numel()
                if max_val.item() > 1e-6 and num_elements > 0:
                    normalized_tensor = (
                        torch.ones_like(aggregated_tensor) / num_elements
                    )
                else:
                    normalized_tensor = torch.zeros_like(aggregated_tensor)
            else:
                normalized_tensor = (aggregated_tensor - min_val) / (range_val + 1e-8)

        # Return the result with only the target variable
        return ProbabilisticTensor(normalized_tensor, vars=[var], normalized=True)

    @cache_action.__get__(object)
    def best(self, var="x1", among=None, alpha=0.5, tau=None):
        """Restricted superlative: rank the entities bound to ``var`` by this measure,
        among the entities in ``among`` only.

        "The closest round object on my right" is
        ``closeness.best("x1", among=round & right)``: ``among`` decides which entities
        compete, the measure decides which of them wins, and the two never mix.
        Eligibility is the alpha-cut ``among >= alpha``, or the soft step
        ``sigmoid((among - alpha) / tau)`` when ``tau`` is given. Other variables of
        the measure or of ``among`` are projected out with exists; only objects bind.

        Returns a formula over ``var``: 1.0 for the best eligible entity, the measure
        relative to the best for the other eligible ones, 0 for the rest (all 0 when
        no entity is eligible).
        """
        def project(t, vs, name):
            if var not in vs:
                if t.dim() == 1 and len(vs) <= 1:
                    return t
                raise ValueError(f"{name} has variables {vs}; it must range over {var!r}")
            others = tuple(i for i, v in enumerate(vs) if v != var)
            return t.amax(dim=others) if others else t

        measure = project(torch.nan_to_num(self.tensor, nan=0.0), list(self.vars), "the measure")
        measure = torch.clamp(measure, min=0.0)
        if among is None:
            eligible = torch.ones_like(measure)
        else:
            if hasattr(among, "to_tensor") and not isinstance(among, ProbabilisticTensor):
                among = among.to_tensor()
            if not isinstance(among, ProbabilisticTensor):
                among = ProbabilisticTensor(torch.as_tensor(among, dtype=measure.dtype), vars=[var])
            s = project(torch.nan_to_num(among.tensor, nan=0.0), list(among.vars), "among").to(measure.device)
            if s.shape != measure.shape:
                raise ValueError(f"among covers {tuple(s.shape)} entities, the measure {tuple(measure.shape)}")
            eligible = (s >= alpha).to(measure.dtype) if tau is None else torch.sigmoid((s - alpha) / tau)
        k = BINDABLE_ENTITIES.get()
        if k is not None and 0 < k < measure.shape[0]:
            eligible = torch.cat([eligible[:k], torch.zeros_like(eligible[k:])])
        scores = measure * eligible
        top = scores.max() if scores.numel() else scores.new_tensor(0.0)
        out = scores / top if top > 0 else torch.zeros_like(scores)
        return ProbabilisticTensor(out, vars=[var])

    @cache_action.__get__(object)
    def mean(self, var=None, dim=-1):
        if var is not None:
            var_index = self.vars.index(var)
            return ProbabilisticTensor(self.tensor.mean(dim=var_index))
        return ProbabilisticTensor(self.tensor.mean(dim=dim))
