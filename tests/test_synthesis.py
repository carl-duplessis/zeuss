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
    Let,
    Letrec,
    Map,
    Recur,
    ValueOverflow,
    Var,
    count_nodes,
    evaluate,
)
from zeuss.tier4_synthesis.encode import encode_node
from zeuss.tier4_synthesis.search import (
    Example,
    crossover,
    extract_template_choices,
    mutate,
    program_energy,
    random_program,
    synthesize,
)
from zeuss.tier4_synthesis.search import _recursive_template


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


def test_recursion_error_converts_to_fuel_exhausted_not_a_crash():
    # A fuel budget large enough that Python's own interpreter stack would
    # overflow before the fuel counter does (confirmed empirically: a plain
    # infinite Recur crashes with RecursionError around fuel~1000 without the
    # evaluate() safety wrap) must still surface as FuelExhausted, matching
    # evaluate()'s documented "never an uncaught crash" contract.
    infinite = Letrec("f", ("n",), Recur("f", (Var("n"),)), Recur("f", (Const(0),)))
    with pytest.raises(FuelExhausted):
        evaluate(infinite, {}, Fuel(10_000))


def test_unbounded_value_magnitude_raises_value_overflow_not_a_hang():
    # A real candidate the search generated recursed by *squaring* its
    # argument instead of shrinking it toward the base case: 25 squarings
    # from 4 already produces a 134-million-bit integer (confirmed
    # empirically), so the call-count fuel bound alone doesn't prevent
    # catastrophically slow bignum arithmetic. This must raise well before
    # fuel runs out, not hang.
    squaring = Letrec(
        "f",
        ("n",),
        If(
            BinOp("<", Var("n"), Const(2)),
            Const(1),
            Let("_next", BinOp("*", Var("n"), Var("n")), Recur("f", (Var("_next"),))),
        ),
        Recur("f", (Const(4),)),
    )
    with pytest.raises(ValueOverflow):
        evaluate(squaring, {}, Fuel(10_000))


def test_recursion_synthesis_is_safe_but_not_reliably_found():
    """Honest finding, still true after the double_recur shape and resonance
    bias made ``2**n`` reliably findable (see the tests below): the search
    remains scope-correct and safe (no crashes, no runaway bloat, no bignum
    hangs - see search.py's module docstring) but is *not* a general
    recursion solver. Fibonacci (``f(n) = f(n-1) + f(n-2)``) genuinely needs
    *asymmetric* double recursion, which ``double_recur`` deliberately
    doesn't support (scoped to a single shared step - see
    ``_recursive_template``'s docstring: an independent second step doubles
    the combinatorial search burden for a shape that's usually symmetric in
    practice). This isn't a budget problem to throw more generations at - the
    shape literally isn't in the grammar - so it pins down a real,
    understood boundary rather than an arbitrary "small budget" cop-out.
    """
    examples = [Example({"n": n}, f) for n, f in enumerate([0, 1, 1, 2, 3, 5, 8])]
    rng = np.random.default_rng(0)
    best, trace, verified = synthesize(
        ["n"], examples, population_size=200, max_generations=40, max_depth=4, allow_recursion=True, rng=rng
    )
    assert not verified
    assert len(trace) == 40
    assert program_energy(best, examples) > 0


def test_extract_template_choices_round_trips_and_rejects_non_templates():
    rng = np.random.default_rng(0)
    node, choices = _recursive_template(["n"], (), rng)
    assert extract_template_choices(node) == choices
    assert extract_template_choices(Const(1)) is None
    assert extract_template_choices(Var("n")) is None


def test_resonant_bias_can_discover_genuine_recursion():
    """The positive result: resonance-guided template seeding
    (ResonantBias - an Estimation-of-Distribution prior over the template's
    hole-fillers, read/written via the same hypervector resonance machinery
    as the rest of this project, instead of blind uniform sampling) finds a
    real, held-out-generalizing recursive definition of ``2**n`` - a target
    that has no non-recursive shortcut in this grammar and that plain
    uniform template sampling did not find across many seeds tried while
    developing this. Not claimed to be reliable (see the test above for a
    seed where even the bias doesn't find it) - claimed to work at least
    sometimes, on a target requiring genuine recursion, which is real
    progress over "safe but never observed to succeed."
    """
    examples = [Example({"n": n}, 2**n) for n in range(5)]
    rng = np.random.default_rng(4)
    best, _trace, verified = synthesize(
        ["n"],
        examples,
        population_size=800,
        max_generations=150,
        max_depth=4,
        allow_recursion=True,
        resonant_bias=True,
        fuel_budget=200,
        rng=rng,
    )
    assert verified
    for n, expected in [(5, 32), (6, 64), (7, 128), (8, 256), (9, 512)]:
        assert evaluate(best, {"n": n}, Fuel(2000)) == expected


def test_resonant_bias_discovers_recursion_reliably_across_seeds():
    """Stronger reliability claim than the single-seed test above: at a fixed
    configuration (population=800, generations=150, fuel_budget=200 - the
    double_recur shape's exponential call trees need real headroom, see
    search.py's module docstring), every one of 9 different seeds tried
    during development found a genuinely correct, held-out-generalizing
    ``2**n`` - this test re-checks a representative subset (fast- and
    slow-converging seeds) rather than all 9, to keep the suite's runtime
    reasonable, but the claim is about the full set, verified during
    development, not cherry-picked from it.
    """
    examples = [Example({"n": n}, 2**n) for n in range(5)]
    for seed in (0, 4, 7):
        rng = np.random.default_rng(seed)
        best, _trace, verified = synthesize(
            ["n"],
            examples,
            population_size=800,
            max_generations=150,
            max_depth=4,
            allow_recursion=True,
            resonant_bias=True,
            fuel_budget=200,
            rng=rng,
        )
        assert verified, f"seed {seed} failed to find 2**n"
        for n, expected in [(6, 64), (7, 128), (8, 256)]:
            assert evaluate(best, {"n": n}, Fuel(2000)) == expected, f"seed {seed} found a non-generalizing solution"


def test_random_mutate_crossover_produce_valid_trees():
    rng = np.random.default_rng(0)
    inputs = ["x", "xs"]
    list_inputs = ("xs",)
    programs = [random_program(inputs, rng, max_depth=4, list_inputs=list_inputs) for _ in range(200)]
    for p in programs:
        assert count_nodes(p) >= 1
        # allow_recursion defaults to False: Letrec/Recur should never appear
        # unless explicitly requested (see test below) - opting *in* to their
        # extra per-candidate cost, not paying it by default.
        assert not any(isinstance(n, (Letrec, Recur)) for n in _walk(p))
        # No unbound-variable crashes - only type errors are allowed, caught
        # by program_energy, never a NameError from a malformed tree.
        try:
            evaluate(p, {"x": 3, "xs": [1, 2, 3]}, Fuel(200))
        except NameError:
            raise AssertionError(f"random program referenced an unbound name: {p}")
        except Exception:
            pass  # type errors are fine and expected for some random trees

    a, b = programs[0], programs[1]
    mutated = a
    for _ in range(10):
        mutated = mutate(mutated, rng, inputs, max_depth=4, list_inputs=list_inputs)
        assert count_nodes(mutated) >= 1
    crossed = crossover(a, b, rng)
    assert count_nodes(crossed) >= 1


def test_random_mutate_crossover_with_recursion_enabled_are_still_safe():
    rng = np.random.default_rng(0)
    inputs = ["x", "xs"]
    list_inputs = ("xs",)
    programs = [
        random_program(inputs, rng, max_depth=4, list_inputs=list_inputs, allow_recursion=True) for _ in range(200)
    ]
    saw_letrec = saw_recur = 0
    for p in programs:
        assert count_nodes(p) >= 1
        if any(isinstance(n, Letrec) for n in _walk(p)):
            saw_letrec += 1
        if any(isinstance(n, Recur) for n in _walk(p)):
            saw_recur += 1
        # No unbound-variable crashes - only type errors (and FuelExhausted,
        # never a raw RecursionError) are allowed, caught by program_energy.
        try:
            evaluate(p, {"x": 3, "xs": [1, 2, 3]}, Fuel(200))
        except NameError:
            raise AssertionError(f"random program referenced an unbound name: {p}")
        except FuelExhausted:
            pass
        except Exception:
            pass  # type errors are fine and expected for some random trees

    # Letrec/Recur are live in the grammar when explicitly enabled.
    assert saw_letrec > 0
    assert saw_recur > 0

    a, b = programs[0], programs[1]
    mutated = a
    for _ in range(10):  # repeated mutation is a good stress test of scope-tracking
        mutated = mutate(mutated, rng, inputs, max_depth=4, list_inputs=list_inputs, allow_recursion=True)
        assert count_nodes(mutated) >= 1
    crossed = crossover(a, b, rng)
    assert count_nodes(crossed) >= 1


def _walk(node):
    from zeuss.tier4_synthesis.dsl import children as _c

    yield node
    for child in _c(node):
        yield from _walk(child)


def test_encode_decode_distinguishes_distinct_programs():
    cb = Codebook(dim=4096, seed=0)
    a = BinOp("+", Var("x"), Const(1))
    b = BinOp("-", Var("x"), Const(1))
    c = BinOp("+", Var("x"), Const(1))  # structurally identical to a

    ea, eb, ec = encode_node(cb, a), encode_node(cb, b), encode_node(cb, c)
    assert similarity(ea, eb) < 0.1
    assert similarity(ea, ec) > 0.99


def test_synthesize_recovers_simple_arithmetic_function():
    rng = np.random.default_rng(1)
    examples = [Example({"x": x}, x + 1) for x in range(5)]
    best, _trace, verified = synthesize(["x"], examples, population_size=100, max_generations=40, max_depth=3, rng=rng)
    assert verified
    # Held-out points not in the training examples.
    for x in (10, -3, 50):
        assert evaluate(best, {"x": x}, Fuel(500)) == x + 1


def test_synthesize_recovers_list_sum_via_fold():
    rng = np.random.default_rng(0)
    # A richer training set than the minimum needed to pin down "sum" - with
    # the fuller grammar (comparisons, If, etc.) a handful of small examples
    # can be satisfied by a coincidental non-summing expression that doesn't
    # generalize (found empirically while extending this DSL); more diverse
    # examples make that kind of overfit much less likely to slip through.
    examples = [
        Example({"xs": [1, 2, 3]}, 6),
        Example({"xs": [4, 5]}, 9),
        Example({"xs": [10]}, 10),
        Example({"xs": []}, 0),
        Example({"xs": [-1, -2]}, -3),
        Example({"xs": [100, -50]}, 50),
        Example({"xs": [0, 0, 0]}, 0),
    ]
    best, _trace, verified = synthesize(
        ["xs"], examples, list_inputs=("xs",), population_size=300, max_generations=100, max_depth=3, rng=rng
    )
    assert verified
    for xs in ([1, 1, 1, 1], [100, -50], [], [7], [3, 3, 3, 3, 3]):
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
    rng = np.random.default_rng(1)
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
