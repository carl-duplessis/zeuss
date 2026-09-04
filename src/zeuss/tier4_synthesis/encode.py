"""Structural hypervector encoding of DSL programs.

Recursive role-filler binding: leaf nodes are codebook symbols; compound
nodes bind an ``OP:<label>`` role wave to the bundle of their (positionally
role-bound) children's encodings - the exact same bind/bundle recursion
:mod:`zeuss.tier3_logic.ontology` already uses for triples, just applied to a
tree instead of a flat triple.
"""
from __future__ import annotations

from ..tier2_substrate.hypervectors import Codebook, bind, bundle
from .dsl import (
    BinOp,
    Const,
    Filter,
    Fold,
    If,
    Index,
    Length,
    Let,
    Letrec,
    ListLit,
    Map,
    Node,
    Recur,
    UnaryOp,
    Var,
    children,
)


def _op_label(node: Node) -> str:
    if isinstance(node, BinOp):
        return f"BinOp:{node.op}"
    if isinstance(node, UnaryOp):
        return f"UnaryOp:{node.op}"
    if isinstance(node, If):
        return "If"
    if isinstance(node, Let):
        return f"Let:{node.name}"
    if isinstance(node, Fold):
        return f"Fold:{node.var_acc}:{node.var_item}"
    if isinstance(node, Length):
        return "Length"
    if isinstance(node, Index):
        return "Index"
    if isinstance(node, Map):
        return f"Map:{node.var_item}"
    if isinstance(node, Filter):
        return f"Filter:{node.var_item}"
    if isinstance(node, Letrec):
        return f"Letrec:{node.name}:{','.join(node.params)}"
    if isinstance(node, Recur):
        return f"Recur:{node.name}"
    if isinstance(node, ListLit):
        return "ListLit"
    return type(node).__name__


def encode_node(codebook: Codebook, node: Node):
    """Encode ``node`` as one hypervector (structural, recursive)."""
    if isinstance(node, Const):
        return codebook.symbol(f"CONST:{node.value!r}")
    if isinstance(node, Var):
        return codebook.symbol(f"VAR:{node.name}")

    kids = children(node)
    if not kids:
        return codebook.symbol(f"LEAF:{type(node).__name__}")

    op_wave = codebook.symbol(f"OP:{_op_label(node)}")
    bound_children = [
        bind(codebook.symbol(f"ROLE:{i}"), encode_node(codebook, child)) for i, child in enumerate(kids)
    ]
    return bind(op_wave, bundle(bound_children))
