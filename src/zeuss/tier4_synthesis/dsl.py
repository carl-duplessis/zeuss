"""A closed, total expression DSL: arithmetic/boolean expressions, lists,
conditionals, bounded loops (structural folds), and bounded recursion
(fuel-limited).

"Total" is load-bearing: every construct that could loop forever carries an
explicit, finite bound, so :func:`evaluate` is guaranteed to terminate without
a wall-clock timeout hack. ``Fold`` is bounded by the length of the list it
folds over (a structural/functional loop, not a ``while``); ``Letrec``/
``Recur`` bounded recursion is bounded by an explicit fuel counter that every
recursive call and fold iteration spends from, raising :class:`FuelExhausted`
(never an uncaught crash) if it runs out. This is a pure-Python tree
interpreter - no ``eval``/``exec`` of untrusted strings.

Honest scoping note: the mutation/crossover search in :mod:`.search` only
generates and mutates ``Const``/``Var``/``ListLit``/``BinOp``/``UnaryOp``/
``If``/``Let``/``Fold`` programs - loops, conditionals, and lists are within
the auto-search's reach. ``Letrec``/``Recur`` (recursion) is fully supported
by this interpreter and directly testable on hand-built programs, but the
search does not attempt to *synthesize* new recursive definitions from
scratch: safely generating well-scoped recursive candidates via random
mutation is a substantially harder search-space design problem than the
loop/conditional/list constructs this phase targets, and claiming otherwise
would not be honest.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union


class FuelExhausted(Exception):
    """Raised when a bounded-recursion/fold computation exceeds its fuel budget."""


@dataclass(frozen=True)
class Const:
    value: object  # int | bool


@dataclass(frozen=True)
class Var:
    name: str


@dataclass(frozen=True)
class ListLit:
    items: tuple["Node", ...]


@dataclass(frozen=True)
class BinOp:
    op: str  # + - * // == < and or
    left: "Node"
    right: "Node"


@dataclass(frozen=True)
class UnaryOp:
    op: str  # - not
    operand: "Node"


@dataclass(frozen=True)
class If:
    cond: "Node"
    then: "Node"
    orelse: "Node"


@dataclass(frozen=True)
class Let:
    name: str
    value: "Node"
    body: "Node"


@dataclass(frozen=True)
class Fold:
    """A structural (guaranteed-terminating) loop over a list."""

    list_expr: "Node"
    init: "Node"
    var_acc: str
    var_item: str
    body: "Node"


@dataclass(frozen=True)
class Letrec:
    """Binds ``name`` as a recursive function visible inside both ``body``
    (its own definition, so it can ``Recur`` into itself) and ``in_expr``."""

    name: str
    params: tuple[str, ...]
    body: "Node"
    in_expr: "Node"


@dataclass(frozen=True)
class Recur:
    """A call to a ``Letrec``-bound recursive function."""

    name: str
    args: tuple["Node", ...]


@dataclass(frozen=True)
class Closure:
    params: tuple[str, ...]
    body: "Node"


Node = Union[Const, Var, ListLit, BinOp, UnaryOp, If, Let, Fold, Letrec, Recur]


class Fuel:
    """A mutable fuel counter threaded through :func:`evaluate`."""

    __slots__ = ("remaining",)

    def __init__(self, remaining: int) -> None:
        self.remaining = remaining

    def spend(self, amount: int = 1) -> None:
        self.remaining -= amount
        if self.remaining < 0:
            raise FuelExhausted()


def _apply_binop(op: str, a, b):
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "//":
        if b == 0:
            raise ZeroDivisionError("division by zero")
        return int(a) // int(b)
    if op == "==":
        return a == b
    if op == "<":
        return a < b
    if op == "and":
        return bool(a) and bool(b)
    if op == "or":
        return bool(a) or bool(b)
    raise ValueError(f"unknown binary op: {op}")


def _apply_unaryop(op: str, v):
    if op == "-":
        return -v
    if op == "not":
        return not v
    raise ValueError(f"unknown unary op: {op}")


def evaluate(node: Node, env: dict, fuel: Fuel):
    """Evaluate ``node`` under ``env``, spending ``fuel`` on every recursive
    call and fold iteration. Raises :class:`FuelExhausted` rather than
    looping forever or timing out."""
    if isinstance(node, Const):
        return node.value
    if isinstance(node, Var):
        if node.name not in env:
            raise NameError(f"unbound variable: {node.name}")
        return env[node.name]
    if isinstance(node, ListLit):
        return [evaluate(item, env, fuel) for item in node.items]
    if isinstance(node, BinOp):
        a = evaluate(node.left, env, fuel)
        b = evaluate(node.right, env, fuel)
        return _apply_binop(node.op, a, b)
    if isinstance(node, UnaryOp):
        return _apply_unaryop(node.op, evaluate(node.operand, env, fuel))
    if isinstance(node, If):
        branch = node.then if evaluate(node.cond, env, fuel) else node.orelse
        return evaluate(branch, env, fuel)
    if isinstance(node, Let):
        value = evaluate(node.value, env, fuel)
        return evaluate(node.body, {**env, node.name: value}, fuel)
    if isinstance(node, Fold):
        items = evaluate(node.list_expr, env, fuel)
        acc = evaluate(node.init, env, fuel)
        for item in items:
            fuel.spend()
            acc = evaluate(node.body, {**env, node.var_acc: acc, node.var_item: item}, fuel)
        return acc
    if isinstance(node, Letrec):
        new_env = dict(env)
        new_env[node.name] = Closure(node.params, node.body)
        return evaluate(node.in_expr, new_env, fuel)
    if isinstance(node, Recur):
        fuel.spend()
        closure = env.get(node.name)
        if not isinstance(closure, Closure):
            raise NameError(f"{node.name} is not a recursive function in scope")
        args = [evaluate(a, env, fuel) for a in node.args]
        call_env = dict(env)
        call_env.update(zip(closure.params, args))
        return evaluate(closure.body, call_env, fuel)
    raise TypeError(f"unknown node type: {type(node)!r}")


def children(node: Node) -> list["Node"]:
    """Direct child nodes, in evaluation order - the shared traversal used by
    :mod:`.encode` and :mod:`.search`'s mutation/crossover."""
    if isinstance(node, ListLit):
        return list(node.items)
    if isinstance(node, BinOp):
        return [node.left, node.right]
    if isinstance(node, UnaryOp):
        return [node.operand]
    if isinstance(node, If):
        return [node.cond, node.then, node.orelse]
    if isinstance(node, Let):
        return [node.value, node.body]
    if isinstance(node, Fold):
        return [node.list_expr, node.init, node.body]
    if isinstance(node, Letrec):
        return [node.body, node.in_expr]
    if isinstance(node, Recur):
        return list(node.args)
    return []  # Const, Var: leaves


def rebuild(node: Node, new_children: list["Node"]) -> "Node":
    """Reconstruct ``node`` with its children replaced by ``new_children``
    (same order as :func:`children`)."""
    if isinstance(node, ListLit):
        return ListLit(tuple(new_children))
    if isinstance(node, BinOp):
        return BinOp(node.op, new_children[0], new_children[1])
    if isinstance(node, UnaryOp):
        return UnaryOp(node.op, new_children[0])
    if isinstance(node, If):
        return If(new_children[0], new_children[1], new_children[2])
    if isinstance(node, Let):
        return Let(node.name, new_children[0], new_children[1])
    if isinstance(node, Fold):
        return Fold(new_children[0], new_children[1], node.var_acc, node.var_item, new_children[2])
    if isinstance(node, Letrec):
        return Letrec(node.name, node.params, new_children[0], new_children[1])
    if isinstance(node, Recur):
        return Recur(node.name, tuple(new_children))
    return node  # Const, Var: no children


def count_nodes(node: Node) -> int:
    return 1 + sum(count_nodes(c) for c in children(node))


def pretty(node: Node) -> str:
    """A readable expression string - not meant to be re-parsed, just read."""
    if isinstance(node, Const):
        return repr(node.value)
    if isinstance(node, Var):
        return node.name
    if isinstance(node, ListLit):
        return "[" + ", ".join(pretty(i) for i in node.items) + "]"
    if isinstance(node, BinOp):
        return f"({pretty(node.left)} {node.op} {pretty(node.right)})"
    if isinstance(node, UnaryOp):
        return f"({node.op} {pretty(node.operand)})"
    if isinstance(node, If):
        return f"(if {pretty(node.cond)} then {pretty(node.then)} else {pretty(node.orelse)})"
    if isinstance(node, Let):
        return f"(let {node.name} = {pretty(node.value)} in {pretty(node.body)})"
    if isinstance(node, Fold):
        return (
            f"(fold {node.var_acc},{node.var_item} in {pretty(node.list_expr)} "
            f"from {pretty(node.init)}: {pretty(node.body)})"
        )
    if isinstance(node, Letrec):
        return f"(letrec {node.name}({', '.join(node.params)}) = {pretty(node.body)} in {pretty(node.in_expr)})"
    if isinstance(node, Recur):
        return f"{node.name}({', '.join(pretty(a) for a in node.args)})"
    return repr(node)
