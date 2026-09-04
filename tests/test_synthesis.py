"""Genetic-programming synthesis over a total DSL, scored by real execution."""
import numpy as np
import pytest

from zeuss.tier2_substrate.hypervectors import Codebook, similarity
from zeuss.tier4_synthesis.dsl import (
    BinOp,
    Const,
    Filter,
    Fold,
    Fuel,
    FuelExhausted,
    If,
    Index,
    Length,
    Letrec,
    Map,
    Recur,
    Var,
    count_nodes,
    evaluate,
)
from zeuss.tier4_synthesis.encode import encode_node
from zeuss.tier4_synthesis.search import Example, crossover, mutate, program_energy, random_program, synthesize


def test_fold_computes_a_bounded_loop():
    fold_sum = Fold(Var("xs"), Const(0), "acc", "item", BinOp("+", Var("acc"), Var("item")))
    assert evaluate(fold_sum, {"xs": [1, 2, 3, 4]}, Fuel(100)) == 10
    assert evaluate(fold_sum, {"xs": []}, Fuel(100)) == 0


def test_list_ops_evaluate_correctly():
    xs = Var("xs")
    assert evaluate(Length(xs), {"xs": [1, 2, 3, 4]}, Fuel(100)) == 4
    assert evaluate(Index(xs, Const(1)), {"xs": [10, 20, 30]}, Fuel(100)) == 20
    with pytest.raises(IndexError):
        evaluate(Index(xs, Const(5)), {"xs": [1, 2]}, Fuel(100))

    doubled = Map(xs, "item", BinOp("*", Var("item"), Const(2)))
    assert evaluate(doubled, {"xs": [1, 2, 3]}, Fuel(100)) == [2, 4, 6]

    evens = Filter(xs, "item", BinOp("==", BinOp("%", Var("item"), Const(2)), Const(0)))
    assert evaluate(evens, {"xs": [1, 2, 3, 4, 5, 6]}, Fuel(100)) == [2, 4, 6]


def test_new_comparison_and_modulo_operators():
    assert evaluate(BinOp("%", Const(7), Const(3)), {}, Fuel(10)) == 1
    assert evaluate(BinOp("<=", Const(3), Const(3)), {}, Fuel(10)) is True
    assert evaluate(BinOp(">=", Const(2), Const(3)), {}, Fuel(10)) is False
    assert evaluate(BinOp("!=", Const(3), Const(4)), {}, Fuel(10)) is True
    assert evaluate(BinOp(">", Const(5), Const(4)), {}, Fuel(10)) is True


def test_bounded_recursion_computes_factorial_and_respects_fuel():
    fact_body = If(
        BinOp("==", Var("n"), Const(0)),
        Const(1),
        BinOp("*", Var("n"), Recur("fact", (BinOp("-", Var("n"), Const(1)),))),
    )
    fact = Letrec("fact", ("n",), fact_body, Recur("fact", (Var("n"),)))
    assert evaluate(fact, {"n": 5}, Fuel(100)) == 120
    assert evaluate(fact, {"n": 0}, Fuel(100)) == 1
    with pytest.raises(FuelExhausted):
        evaluate(fact, {"n": 10_000}, Fuel(50))


def test_random_mutate_crossover_produce_valid_trees():
    rng = np.random.default_rng(0)
    inputs = ["x", "xs"]
    list_inputs = ("xs",)
    programs = [random_program(inputs, rng, max_depth=4, list_inputs=list_inputs) for _ in range(20)]
    for p in programs:
        assert count_nodes(p) >= 1
        # No unbound-variable crashes - only type errors are allowed, caught
        # by program_energy, never a NameError from a malformed tree.
        try:
            evaluate(p, {"x": 3, "xs": [1, 2, 3]}, Fuel(200))
        except NameError:
            raise AssertionError(f"random program referenced an unbound name: {p}")
        except Exception:
            pass  # type errors are fine and expected for some random trees

    a, b = programs[0], programs[1]
    mutated = mutate(a, rng, inputs, max_depth=4, list_inputs=list_inputs)
    crossed = crossover(a, b, rng)
    assert count_nodes(mutated) >= 1
    assert count_nodes(crossed) >= 1


def test_encode_decode_distinguishes_distinct_programs():
    cb = Codebook(dim=4096, seed=0)
    a = BinOp("+", Var("x"), Const(1))
    b = BinOp("-", Var("x"), Const(1))
    c = BinOp("+", Var("x"), Const(1))  # structurally identical to a

    ea, eb, ec = encode_node(cb, a), encode_node(cb, b), encode_node(cb, c)
    assert similarity(ea, eb) < 0.1
    assert similarity(ea, ec) > 0.99


def test_synthesize_recovers_simple_arithmetic_function():
    rng = np.random.default_rng(0)
    examples = [Example({"x": x}, x + 1) for x in range(5)]
    best, _trace, verified = synthesize(["x"], examples, population_size=100, max_generations=40, max_depth=3, rng=rng)
    assert verified
    # Held-out points not in the training examples.
    for x in (10, -3, 50):
        assert evaluate(best, {"x": x}, Fuel(500)) == x + 1


def test_synthesize_recovers_list_sum_via_fold():
    rng = np.random.default_rng(5)
    examples = [
        Example({"xs": [1, 2, 3]}, 6),
        Example({"xs": [4, 5]}, 9),
        Example({"xs": [10]}, 10),
        Example({"xs": []}, 0),
    ]
    best, _trace, verified = synthesize(
        ["xs"], examples, list_inputs=("xs",), population_size=300, max_generations=100, max_depth=3, rng=rng
    )
    assert verified
    for xs in ([1, 1, 1, 1], [100, -50], [], [7]):
        assert evaluate(best, {"xs": xs}, Fuel(500)) == sum(xs)


def test_synthesize_recovers_length():
    rng = np.random.default_rng(0)
    examples = [
        Example({"xs": [1, 2, 3]}, 3),
        Example({"xs": []}, 0),
        Example({"xs": [5, 5]}, 2),
        Example({"xs": [1]}, 1),
    ]
    best, _trace, verified = synthesize(
        ["xs"], examples, list_inputs=("xs",), population_size=100, max_generations=30, max_depth=2, rng=rng
    )
    assert verified
    for xs in ([1, 2, 3, 4, 5], [], [9]):
        assert evaluate(best, {"xs": xs}, Fuel(500)) == len(xs)


def test_synthesize_recovers_map_doubling():
    rng = np.random.default_rng(0)
    examples = [
        Example({"xs": [1, 2, 3]}, [2, 4, 6]),
        Example({"xs": []}, []),
        Example({"xs": [5]}, [10]),
    ]
    best, _trace, verified = synthesize(
        ["xs"], examples, list_inputs=("xs",), population_size=150, max_generations=60, max_depth=2, rng=rng
    )
    assert verified
    for xs in ([1, 1, 1], [10, -5]):
        assert evaluate(best, {"xs": xs}, Fuel(500)) == [x * 2 for x in xs]


def test_synthesize_recovers_filter_positives():
    rng = np.random.default_rng(0)
    examples = [
        Example({"xs": [1, -2, 3, -4]}, [1, 3]),
        Example({"xs": []}, []),
        Example({"xs": [-1, -2]}, []),
        Example({"xs": [5]}, [5]),
    ]
    best, _trace, verified = synthesize(
        ["xs"], examples, list_inputs=("xs",), population_size=200, max_generations=60, max_depth=2, rng=rng
    )
    assert verified
    for xs in ([-1, 2, -3, 4, 5], [0, 0, 1]):
        assert evaluate(best, {"xs": xs}, Fuel(500)) == [x for x in xs if x > 0]


def test_synthesize_recovers_first_element_via_index():
    rng = np.random.default_rng(1)
    examples = [
        Example({"xs": [7, 8, 9]}, 7),
        Example({"xs": [1]}, 1),
        Example({"xs": [-5, 2]}, -5),
    ]
    best, _trace, verified = synthesize(
        ["xs"], examples, list_inputs=("xs",), population_size=100, max_generations=30, max_depth=2, rng=rng
    )
    assert verified
    for xs in ([42, 1, 2], [100]):
        assert evaluate(best, {"xs": xs}, Fuel(500)) == xs[0]


def test_synthesize_reports_unverified_when_infeasible_within_budget():
    rng = np.random.default_rng(0)
    # x*7 is unreachable with this tiny population/generation/depth budget.
    examples = [Example({"x": x}, x * 7) for x in range(1, 4)]
    best, trace, verified = synthesize(["x"], examples, population_size=20, max_generations=5, max_depth=2, rng=rng)
    assert not verified
    assert program_energy(best, examples) > 0
    assert len(trace) == 5


def test_selection_pressure_rises_across_generations():
    rng = np.random.default_rng(3)
    examples = [Example({"x": x}, x * 7) for x in range(1, 4)]  # deliberately hard -> runs full budget
    _best, trace, _verified = synthesize(["x"], examples, population_size=20, max_generations=10, max_depth=2, rng=rng)
    assert all(a <= b + 1e-9 for a, b in zip(trace, trace[1:]))
