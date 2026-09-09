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
    UnaryOp,
    ValueOverflow,
    Var,
    count_nodes,
    evaluate,
    pretty,
)
from zeuss.tier4_synthesis.encode import encode_node
from zeuss.tier4_synthesis.grammar_bias import (
    GrammarBias,
    collect_production_choices,
    load_grammar_bias,
    production_context,
    save_grammar_bias,
)
from zeuss.tier4_synthesis.motif_bias import MotifArchive, collect_motifs, context_key
from zeuss.tier4_synthesis.resonance_bias import ResonantBias
from zeuss.tier4_synthesis.semantic_bias import (
    and_left_target,
    and_right_target,
    best_atom_for_target,
    boolean_backprop_template,
    grow_boolean_targeted,
    not_target,
    or_left_target,
    or_right_target,
)
from zeuss.tier4_synthesis.search import GRAMMAR_CHOICE_VOCAB
from zeuss.tier4_synthesis.search import (
    Example,
    crossover,
    extract_template_choices,
    mutate,
    program_energy,
    random_program,
    root_shape,
    synthesize,
)
from zeuss.tier4_synthesis.search import _recursive_template
from zeuss.tier4_synthesis.search import (
    BOOL_TEMPLATE_CATEGORIES,
    _bool_template,
    _build_atom_node,
    _build_bool_template_node,
    _mutate_bool_template_hole,
    extract_bool_template_choices,
)


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
    """Honest finding, still true even after ``double_recur`` grew an
    asymmetric ``delta`` (see ``_recursive_template`` and the Fibonacci tests
    below, which *do* find asymmetric solutions on some seeds): the search
    remains scope-correct and safe (no crashes, no runaway bloat, no bignum
    hangs - see search.py's module docstring) but is *not* a reliable general
    recursion solver. At this test's deliberately small budget (population
    200, generations 40, default fuel_budget=60 - too small for
    ``double_recur``'s exponential call trees to pay off even for a correct
    asymmetric candidate), Fibonacci (``f(n) = f(n-1) + f(n-2)``) is not
    found. This is a budget/reliability boundary, not a grammar boundary -
    see ``test_resonant_bias_can_discover_fibonacci`` for a budget at which
    it sometimes succeeds, and its docstring for the honest reliability
    picture across seeds (including a seed that "verifies" against the
    training examples without actually generalizing).
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


def test_extract_template_choices_handles_param_base_case():
    """``base_kind="param"`` (``Var(param)`` as the base case, instead of a
    fixed ``Const``) is what makes a *true* zero-indexed Fibonacci (``F(0)=0``)
    expressible - see ``_recursive_template``'s docstring. Hand-built here
    (rather than relying on a random draw to hit this branch) so the
    extraction round-trip for it is pinned down directly."""
    fib0 = Letrec(
        "fib",
        ("p",),
        If(
            BinOp("<=", Var("p"), Const(1)),
            Var("p"),
            BinOp(
                "+",
                Recur("fib", (BinOp("-", Var("p"), Const(1)),)),
                Recur("fib", (BinOp("-", Var("p"), Const(2)),)),
            ),
        ),
        Recur("fib", (Var("n"),)),
    )
    assert evaluate(fib0, {"n": 8}, Fuel(500)) == 21
    assert extract_template_choices(fib0) == {
        "cmp": "<=",
        "base_const": 1,
        "op": "+",
        "base_kind": "param",
        "step": 1,
        "delta": 1,
        "combine_kind": "double_recur",
    }


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


def test_resonant_bias_can_discover_fibonacci():
    """Asymmetric recursion, positive result: Fibonacci (shifted to
    ``F(1)=F(2)=1`` so the base case is expressible as the template's fixed
    ``Const(base_val)``, not ``Var(param)`` - see ``_recursive_template``'s
    docstring) needs ``double_recur`` with an *asymmetric* ``delta`` (two
    different recursive calls, ``f(p-step)`` and ``f(p-step-delta)``), unlike
    ``2**n``'s symmetric ``delta=0`` shape. ``delta``'s prior is deliberately
    *not* resonance-biased (see :class:`ResonantBias`'s and ``synthesize``'s
    docstrings for why a structural choice risks premature convergence) -
    instead it anneals on a stagnation-triggered schedule
    (``delta_p1_start``/``delta_p1_max``/``delta_p1_stagnation_growth``) so a
    run only pays the extra asymmetric-search cost once it's actually stuck.

    Honest reliability picture, measured across seeds at this same
    configuration: seeds 0 and 1 both find a genuinely correct,
    held-out-generalizing Fibonacci (this test re-checks seed 1, the faster
    of the two, to keep the suite's runtime reasonable). Seed 4 does not
    verify within this budget. Seed 7 does "verify" (all six training
    examples match) but with a degenerate, coincidental expression that does
    *not* generalize to held-out n - a real, useful reminder that
    ``verified`` means "matched the given examples," never "proven correct,"
    exactly as this module's honesty statement already says (see
    ``synth.py``). This test only asserts the genuine-generalization case;
    it does not claim Fibonacci is as reliably found as ``2**n`` is.

    A follow-up was tried to fix seeds 4 and 7 specifically: adding one more
    training example (``n=7``, 8 examples total instead of 7). That *did* fix
    both - seed 4 verified and generalized, seed 7 stopped overfitting - but
    it broke seed 1 (which had generalized fine on 7 examples) instead. This
    is a genuine whack-a-mole, not a fixable-with-more-data problem: unlike
    ``2**n``'s resolved 9/9, no single fixed example set was found that gets
    every tried seed to generalize. Recorded here rather than "fixed" by
    picking whichever example count happens to pass this particular test.
    """
    fib = [0, 1, 1, 2, 3, 5, 8]
    examples = [Example({"n": n}, f) for n, f in enumerate(fib)]
    rng = np.random.default_rng(1)
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
    for n, expected in [(7, 13), (8, 21), (9, 34)]:
        assert evaluate(best, {"n": n}, Fuel(2000)) == expected


def test_resonant_bias_can_discover_zero_indexed_fibonacci():
    """``base_kind="param"`` (see ``_recursive_template`` and the extraction
    test above) makes a *true* zero-indexed Fibonacci (``F(0)=0, F(1)=1``)
    directly discoverable, without reindexing the target to ``F(1)=F(2)=1``
    to work around the grammar the way the test above still has to (that
    test predates ``base_kind`` and is kept as-is since it's still a valid,
    still-useful torture test of the ``base_kind="const"`` branch).
    """
    fib = [0, 1, 1, 2, 3, 5, 8]
    examples = [Example({"n": n}, f) for n, f in enumerate(fib)]
    rng = np.random.default_rng(0)
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
    for n, expected in [(7, 13), (8, 21), (9, 34)]:
        assert evaluate(best, {"n": n}, Fuel(2000)) == expected


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
    """A richer training set than the minimum needed to pin down "sum" - with
    the fuller grammar (comparisons, If, etc.) a handful of small examples
    can be satisfied by a coincidental non-summing expression that doesn't
    generalize (found empirically while extending this DSL); more diverse
    examples make that kind of overfit much less likely to slip through.

    Reliability note: seed 3 of an 8-seed sweep failed to find anything at
    all (docs/ROADMAP.md v0.22/v0.23) - diagnosed as converging on a non-
    ``Fold`` ``Index``/``Map`` expression and burning its whole budget
    stuck there, a failure neither richer examples nor a larger budget
    fixed without relocating it to a different seed. It was left at 7/8 by
    deliberate choice rather than tuned around. v0.24's fold template
    (``_fold_template``, added for ``sum_of_squares_via_fold``) fixed it as
    a side effect, confirmed directly rather than assumed: "sum" is exactly
    ``combine_op="+", item_kind="identity"``, one of the template's three
    direct draws, so seed 3's population now has a structural escape route
    it never had before. Re-swept at 8/8 after landing v0.24.
    """
    rng = np.random.default_rng(0)
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


def test_synthesize_recovers_sum_of_squares_via_fold():
    """The list-op counterpart of ``test_resonant_bias_can_discover_genuine_
    recursion``: a fold shape the search could not reliably find at all
    before search.py grew a fold template. The v0.22 audit (docs/ROADMAP.md)
    measured this target at 3/8 seeds, and *worse* with a larger budget
    (1/8) - a parsimony-pressure explanation was tested and refuted, leaving
    it an open capability limit. v0.24 added ``_fold_template`` (the
    ``Fold``-shape counterpart to ``_recursive_template`` -
    ``acc COMBINE_OP transform(item)`` with a resonance-biased ``cmp``/
    ``const`` for the comparison-transform case), and it fixed this
    target outright: 8/8 across the same seed sweep, each converging in a
    handful of generations, since the correct structure is now a direct
    template match rather something ordinary blind growth had to stumble
    onto.
    """
    rng = np.random.default_rng(1)
    examples = [
        Example({"xs": [1, 2, 3]}, 14),
        Example({"xs": [4, 5]}, 41),
        Example({"xs": [10]}, 100),
        Example({"xs": []}, 0),
        Example({"xs": [-1, -2]}, 5),
        Example({"xs": [0, 0, 0]}, 0),
    ]
    best, _trace, verified = synthesize(
        ["xs"], examples, list_inputs=("xs",), population_size=300, max_generations=100, max_depth=3, rng=rng
    )
    assert verified
    for xs in ([1, 1, 1, 1], [3, -3], [], [7], [2, 2, 2]):
        assert evaluate(best, {"xs": xs}, Fuel(500)) == sum(x * x for x in xs)


def test_extract_bool_template_choices_disambiguates_and2_from_and_or3():
    """and2 and and_or3 are both top-level `and` BinOps - the disambiguation
    (does the right child look like a single atom, or an or-of-two-atoms)
    is the one genuinely novel/risky piece of logic this template adds over
    the recursion/fold templates' simpler extraction, so it gets a direct
    unit test rather than relying only on end-to-end target tests (which is
    otherwise this module's own convention - neither the recursion nor fold
    template has unit-level round-trip tests of their own)."""
    and2 = _build_bool_template_node(
        "x", {"shape_kind": "and2", "atoms": [{"modulus": 4, "cmp": "==", "const": 0}, {"modulus": 3, "cmp": "!=", "const": 1}]}
    )
    and_or3 = _build_bool_template_node(
        "x",
        {
            "shape_kind": "and_or3",
            "atoms": [
                {"modulus": 4, "cmp": "==", "const": 0},
                {"modulus": 100, "cmp": "!=", "const": 0},
                {"modulus": 400, "cmp": "==", "const": 0},
            ],
        },
    )
    or2 = _build_bool_template_node(
        "x", {"shape_kind": "or2", "atoms": [{"modulus": 2, "cmp": "==", "const": 0}, {"modulus": 3, "cmp": "==", "const": 0}]}
    )
    or_and3 = _build_bool_template_node(
        "x",
        {
            "shape_kind": "or_and3",
            "atoms": [
                {"modulus": 400, "cmp": "==", "const": 0},
                {"modulus": 4, "cmp": "==", "const": 0},
                {"modulus": 100, "cmp": "!=", "const": 0},
            ],
        },
    )
    assert extract_bool_template_choices(and2)["shape_kind"] == "and2"
    assert extract_bool_template_choices(and_or3)["shape_kind"] == "and_or3"
    assert extract_bool_template_choices(or2)["shape_kind"] == "or2"
    assert extract_bool_template_choices(or_and3)["shape_kind"] == "or_and3"


def test_extract_bool_template_choices_rejects_non_matching_and_mixed_variable_shapes():
    plain_arithmetic = BinOp("+", Var("x"), Const(1))
    assert extract_bool_template_choices(plain_arithmetic) is None

    # Structurally and/or-shaped, but the two atoms reference different
    # variables - not a shape this template reinforces or refines.
    mixed = BinOp(
        "and",
        BinOp("==", BinOp("%", Var("x"), Const(4)), Const(0)),
        BinOp("==", BinOp("%", Var("y"), Const(4)), Const(0)),
    )
    assert extract_bool_template_choices(mixed) is None


def test_and_or3_evaluates_to_exactly_the_leap_year_rule():
    """Traced by hand against the real Gregorian rule while designing this
    template (docs/ROADMAP.md); checked directly here, not just asserted in
    prose: and_or3 with (4,"==",0),(100,"!=",0),(400,"==",0) is exactly
    ``year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)``."""
    node = _build_bool_template_node(
        "year",
        {
            "shape_kind": "and_or3",
            "atoms": [
                {"modulus": 4, "cmp": "==", "const": 0},
                {"modulus": 100, "cmp": "!=", "const": 0},
                {"modulus": 400, "cmp": "==", "const": 0},
            ],
        },
    )

    def is_leap(year):
        return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

    for year in (1900, 2000, 2024, 2023, 1996, 1800, 2400, 2020, 1600):
        assert evaluate(node, {"year": year}, Fuel(200)) == is_leap(year)


def test_mutate_bool_template_hole_changes_exactly_one_atom_field():
    rng = np.random.default_rng(0)
    node, choices = _bool_template(["year"], (), rng)
    mutated = _mutate_bool_template_hole(node, rng, None)
    mutated_choices = extract_bool_template_choices(mutated)
    assert mutated_choices["shape_kind"] == choices["shape_kind"]
    diffs = [
        (i, k)
        for i, (before, after) in enumerate(zip(choices["atoms"], mutated_choices["atoms"]))
        for k in ("modulus", "cmp", "const")
        if before[k] != after[k]
    ]
    assert len(diffs) <= 1


def test_synthesize_recovers_leap_year_rule():
    """The Gregorian leap-year rule
    (``year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)``) was the
    first real (not toy) problem this project's synthesis was pointed at,
    and it reliably failed: the search converges to ``not (year % 2)``
    ("is even"), a deceptive local optimum satisfying every training
    example except the real century exceptions, with no smooth gradient to
    the true structure - confirmed by measurement (0/5 seeds verified at
    this budget's generation count halved; richer examples alone made it
    *worse*; population=1000 still converged to the identical "is even"
    energy). ``_bool_template`` (this module's third structural template,
    after recursion and fold) fixes it: this exact target now converges in
    single-digit generations on most seeds. Training examples deliberately
    include several real century-exception years (1900, 1800, 1700, 2100,
    2200, 2300) spanning varied mod-3/5/7 residues, and two real leap years
    that are *also* divisible by 5 (2020, 1980) - both found necessary by
    diagnosing actual coincidental-fit failures during development (a
    single modulus like 5 or 7 can fit just one or two century-exception
    examples by coincidence; real century years differ enough in other
    small moduli that no such shortcut fits them all at once), not assumed
    sufficient in advance. See ``test_leap_year_rule_is_reliable_across_seeds``
    for the multi-seed reliability measurement, and ``docs/ROADMAP.md`` for
    the full history including two configurations that were tried and moved
    the one remaining failure to a different seed rather than eliminating it.
    """
    rng = np.random.default_rng(0)
    examples = [
        Example({"year": y}, is_leap)
        for y, is_leap in [
            (y, y % 4 == 0 and (y % 100 != 0 or y % 400 == 0))
            for y in (
                2023, 2021, 2019, 1999, 2001,
                2024, 1996, 2004, 1988, 2012, 2020, 1980,
                1900, 1800, 1700, 2100, 2200, 2300,
                2000, 1600, 2400,
            )
        ]
    ]
    best, _trace, verified = synthesize(
        ["year"], examples, population_size=300, max_generations=150, max_depth=4,
        allow_bool_template=True, rng=rng,
    )
    assert verified
    for year, expected in [
        (2025, False), (2028, True), (1904, True), (1596, True),
        (1960, True), (2008, True), (1500, False), (3000, False), (2044, True),
    ]:
        assert evaluate(best, {"year": year}, Fuel(200)) == expected


def test_leap_year_rule_is_reliable_across_seeds():
    """Mirrors ``test_resonant_bias_discovers_recursion_reliably_across_seeds``/
    ``test_list_ops_are_reliable_across_seeds``: a target is only claimed
    reliable if it holds across seeds, not because one chosen seed passes.
    Seeds 0-7 are 8/8 at this exact configuration - measured directly, and a
    representative subset rather than every seed tried during development
    for the same reason the recursion reliability test uses one: a wider
    30-seed sweep found one further failure (seed 19, a genuine ``AND(year
    is divisible by 4, year is not divisible by 100)`` local optimum that
    correctly handles every training example except the three div-by-400
    exceptions) - a real, honestly-measured residual gap, not swept under
    the rug, recorded in ``docs/ROADMAP.md`` rather than tuned around (a
    larger population fixed seed 19 but moved the same failure to two
    different seeds instead - the identical whack-a-mole shape this
    project's own history already has several examples of).
    """
    examples = [
        Example({"year": y}, is_leap)
        for y, is_leap in [
            (y, y % 4 == 0 and (y % 100 != 0 or y % 400 == 0))
            for y in (
                2023, 2021, 2019, 1999, 2001,
                2024, 1996, 2004, 1988, 2012, 2020, 1980,
                1900, 1800, 1700, 2100, 2200, 2300,
                2000, 1600, 2400,
            )
        ]
    ]
    held_out = [
        (2025, False), (2028, True), (1904, True), (1596, True),
        (1960, True), (2008, True), (1500, False), (3000, False), (2044, True),
    ]
    for seed in range(8):
        rng = np.random.default_rng(seed)
        best, _trace, verified = synthesize(
            ["year"], examples, population_size=300, max_generations=150, max_depth=4,
            allow_bool_template=True, rng=rng,
        )
        assert verified, f"leap year seed {seed} found nothing"
        for year, expected in held_out:
            assert evaluate(best, {"year": year}, Fuel(200)) == expected, (
                f"leap year seed {seed} verified on a non-generalizing program (year={year})"
            )


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
    # 150/60 was under-provisioned, not merely unlucky: it finds nothing at
    # all (verified=False) on 2 of 8 seeds swept. 400/150 is 8/8 and still
    # runs in well under a second, since a failed search is what is slow
    # here, not a successful one. See docs/ROADMAP.md v0.22.
    best, _trace, verified = synthesize(
        ["xs"], examples, list_inputs=("xs",), population_size=400, max_generations=150, max_depth=2, rng=rng
    )
    assert verified
    for xs in ([1, 1, 1], [10, -5]):
        assert evaluate(best, {"xs": xs}, Fuel(500)) == [x * 2 for x in xs]


def test_synthesize_recovers_filter_positives():
    """The example set deliberately pins down *positive* (``x > 0``), not
    merely *non-negative* (``x >= 0``), and breaks an accidental
    odd/positive correlation - both found empirically, not guessed. The
    original four examples contained no ``0`` at all, leaving ``> 0`` and
    ``>= 0`` indistinguishable on the training set: sweeping 8 seeds, half of
    them "verified" on ``Filter(xs, item, item >= False)`` (i.e. ``>= 0``)
    and then failed the held-out ``[0, 0, 1]``. Worse, in ``[1, -2, 3, -4]``
    the positives are *exactly* the odd values, and one seed duly learned
    ``item % 2`` instead. Adding ``[0]``, ``[0, 2, -5]`` and ``[4, 6]``
    (zeros on both sides of the boundary, plus even positives) takes this
    from 4/8 seeds to 8/8 at the same search budget - a training-data fix,
    with no change to search.py. See ``docs/ROADMAP.md`` v0.22.
    """
    rng = np.random.default_rng(1)
    examples = [
        Example({"xs": [1, -2, 3, -4]}, [1, 3]),
        Example({"xs": []}, []),
        Example({"xs": [-1, -2]}, []),
        Example({"xs": [5]}, [5]),
        Example({"xs": [0]}, []),
        Example({"xs": [0, 2, -5]}, [2]),
        Example({"xs": [4, 6]}, [4, 6]),
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
    # 100/30 fails outright on seed 0 of 8 swept; this test happened to be
    # written against seed 1, which passes - exactly the single-seed luck the
    # v0.22 audit was looking for. 300/100 is 8/8. See docs/ROADMAP.md v0.22.
    best, _trace, verified = synthesize(
        ["xs"], examples, list_inputs=("xs",), population_size=300, max_generations=100, max_depth=2, rng=rng
    )
    assert verified
    for xs in ([42, 1, 2], [100]):
        assert evaluate(best, {"xs": xs}, Fuel(500)) == xs[0]


def test_list_ops_are_reliable_across_seeds():
    """Multi-seed reliability for the list-op targets, mirroring what
    ``test_resonant_bias_discovers_recursion_reliably_across_seeds`` does for
    recursion - and added for the same reason it was.

    Every list-op test above pins down exactly one seed. The v0.22 audit
    (``docs/ROADMAP.md``) swept them across 8 seeds each and found 19 of 64
    runs failing: ``filter_positives`` silently "verified" on a wrong program
    on half its seeds, ``map_doubling`` found nothing on 2, and
    ``first_via_index``'s committed seed happened to be one of the few that
    worked. None of that was visible from single-seed tests - which is the
    actual lesson, and why this test exists: a target is only claimed
    reliable if it holds across seeds, not because one chosen seed passes.

    Deliberately asserts *verified and generalises to held-out input*, never
    just ``verified`` - the audit's most common failure shape was a program
    that matched every training example and still diverged on held-out data
    (see ``synth.py``'s honesty statement: ``verified`` means "matched the
    given examples", never "proven correct").

    ``sum_of_squares_via_fold`` was added later (v0.24, after the fold
    template) rather than at audit time - it's the one target the audit's
    own initial fix attempt didn't touch, since fixing it needed a genuine
    structural addition, not an example-set or budget change (see its own
    committed test's docstring).
    """
    cases = [
        (
            "length",
            [
                Example({"xs": [1, 2, 3]}, 3), Example({"xs": []}, 0),
                Example({"xs": [5, 5]}, 2), Example({"xs": [1]}, 1),
            ],
            dict(population_size=100, max_generations=30, max_depth=2),
            ([1, 2, 3, 4, 5], [], [9]),
            len,
        ),
        (
            "map_doubling",
            [
                Example({"xs": [1, 2, 3]}, [2, 4, 6]), Example({"xs": []}, []),
                Example({"xs": [5]}, [10]),
            ],
            dict(population_size=400, max_generations=150, max_depth=2),
            ([1, 1, 1], [10, -5]),
            lambda xs: [x * 2 for x in xs],
        ),
        (
            "filter_positives",
            [
                Example({"xs": [1, -2, 3, -4]}, [1, 3]), Example({"xs": []}, []),
                Example({"xs": [-1, -2]}, []), Example({"xs": [5]}, [5]),
                Example({"xs": [0]}, []), Example({"xs": [0, 2, -5]}, [2]),
                Example({"xs": [4, 6]}, [4, 6]),
            ],
            dict(population_size=200, max_generations=60, max_depth=2),
            ([-1, 2, -3, 4, 5], [0, 0, 1]),
            lambda xs: [x for x in xs if x > 0],
        ),
        (
            "first_via_index",
            [
                Example({"xs": [7, 8, 9]}, 7), Example({"xs": [1]}, 1),
                Example({"xs": [-5, 2]}, -5),
            ],
            dict(population_size=300, max_generations=100, max_depth=2),
            ([42, 1, 2], [100]),
            lambda xs: xs[0],
        ),
        (
            "sum_of_squares_via_fold",
            [
                Example({"xs": [1, 2, 3]}, 14), Example({"xs": [4, 5]}, 41),
                Example({"xs": [10]}, 100), Example({"xs": []}, 0),
                Example({"xs": [-1, -2]}, 5), Example({"xs": [0, 0, 0]}, 0),
            ],
            dict(population_size=300, max_generations=100, max_depth=3),
            ([1, 1, 1, 1], [3, -3], [], [7], [2, 2, 2]),
            lambda xs: sum(x * x for x in xs),
        ),
    ]
    for name, examples, kwargs, held_out, expected_fn in cases:
        for seed in range(8):
            rng = np.random.default_rng(seed)
            best, _trace, verified = synthesize(
                ["xs"], examples, list_inputs=("xs",), rng=rng, **kwargs
            )
            assert verified, f"{name} seed {seed} found nothing"
            for xs in held_out:
                assert evaluate(best, {"xs": xs}, Fuel(500)) == expected_fn(xs), (
                    f"{name} seed {seed} verified on a non-generalizing program (xs={xs})"
                )


def test_synth_wrapper_exposes_allow_bool_template():
    """`tier4_synthesis/synth.py` is a curated public wrapper around
    `search.synthesize` that whitelists which parameters it forwards - it
    does not automatically pick up new ones. This was found the hard way,
    not by the 150+ tests in this file (every one of them imports
    `synthesize` from `search` directly, bypassing the wrapper entirely):
    `allow_bool_template` was added to `search.synthesize` and to every
    test/CLI call site that imports it directly, but `cli.py`'s actual
    `zeuss synth` command imports from this wrapper, which had no such
    parameter - `python -m zeuss synth` raised `TypeError: synthesize() got
    an unexpected keyword argument 'allow_bool_template'` despite the full
    test suite being green. Exercises the wrapper directly (not `search`) so
    this class of bug - a new search.py parameter never threaded through the
    curated public entry point - can't silently recur."""
    from zeuss.tier4_synthesis.synth import synthesize as public_synthesize

    def is_leap(year):
        return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

    examples = [Example({"year": y}, is_leap(y)) for y in (2023, 2024, 1900, 2000)]
    rng = np.random.default_rng(0)
    best, _trace, verified = public_synthesize(
        ["year"], examples, population_size=50, max_generations=20, max_depth=4,
        allow_bool_template=True, rng=rng,
    )
    assert best is not None
    assert isinstance(verified, bool)


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


def test_motif_archive_registers_samples_and_evicts():
    """Unit-level check of `MotifArchive` in isolation (see `motif_bias.py`),
    independent of whether the mechanism helps any particular search target
    (see the honest measured result below) - `register`/`sample`/`has`/
    capacity-eviction should behave correctly regardless."""
    cb = Codebook(dim=128, seed=0)
    archive = MotifArchive(ResonantBias(cb, {}), max_motifs_per_context=2)
    ctx = context_key("BinOp", 0, 1)
    assert not archive.has(ctx)
    assert archive.sample(ctx, np.random.default_rng(0)) is None

    a, b, c, d = Const(1), Const(2), Const(3), Const(4)
    archive.register(ctx, a, weight=1.0)
    archive.register(ctx, b, weight=5.0)
    assert archive.has(ctx)
    assert set(archive.bias.categories[ctx]) == {"1", "2"}

    # A low-weight newcomer at capacity should not evict anything - it's
    # worth less than the current lowest-weight entry.
    archive.register(ctx, c, weight=0.1)
    assert set(archive.bias.categories[ctx]) == {"1", "2"}

    # A high-weight newcomer should evict the current lowest-weight entry.
    archive.register(ctx, d, weight=10.0)
    assert set(archive.bias.categories[ctx]) == {"2", "4"}

    sampled = archive.sample(ctx, np.random.default_rng(1))
    assert pretty(sampled) in ("2", "4")


def test_collect_motifs_yields_contexts_matching_grow_sites():
    """`collect_motifs`'s ``(parent_kind, child_slot, depth)`` convention
    must match what `_grow` itself threads through its own recursion (see
    `_grow`'s ``parent_kind="BinOp", child_slot=0/1`` calls) - otherwise a
    motif registered here would never be addressable at the grow-site that
    could actually reuse it."""
    tree = BinOp("+", Const(1), Const(2))
    found = dict(collect_motifs(tree))
    assert found[context_key("root", 0, 0)] == tree
    assert found[context_key("BinOp", 0, 1)] == Const(1)
    assert found[context_key("BinOp", 1, 1)] == Const(2)


def test_motif_bias_is_a_true_noop_by_default():
    """`allow_motif_bias=False` (the default) must leave every existing
    caller's behavior bit-for-bit unchanged: the new grow-site check added to
    `_grow` is guarded to never spend an rng draw when disabled, so the exact
    same seed must produce the exact same tree whether or not the new
    parameters are passed at all. Locks this in directly rather than relying
    solely on "the rest of the suite still passes" (true, but none of those
    152 other tests actually exercise the new parameters, so they can't
    catch a future refactor that moves the guard - e.g. computing
    `context_key`/checking `motif_archive.has(...)` before checking
    `allow_motif_bias` itself)."""
    program_a = random_program(["x"], np.random.default_rng(7), max_depth=4)
    program_b = random_program(
        ["x"], np.random.default_rng(7), max_depth=4, allow_motif_bias=False, motif_archive=None
    )
    assert pretty(program_a) == pretty(program_b)

    examples = [Example({"x": x}, x + 1) for x in range(4)]
    best_c, trace_c, verified_c = synthesize(
        ["x"], examples, population_size=30, max_generations=10, max_depth=3, rng=np.random.default_rng(2)
    )
    best_d, trace_d, verified_d = synthesize(
        ["x"], examples, population_size=30, max_generations=10, max_depth=3,
        allow_motif_bias=False, rng=np.random.default_rng(2),
    )
    assert pretty(best_c) == pretty(best_d)
    assert trace_c == trace_d
    assert verified_c == verified_d


def test_synth_wrapper_exposes_allow_motif_bias():
    """Mirrors `test_synth_wrapper_exposes_allow_bool_template`: guards
    against the same class of bug (a new `search.synthesize` parameter never
    threaded through the curated public wrapper in `synth.py`) recurring for
    the parameters this entry added."""
    from zeuss.tier4_synthesis.synth import synthesize as public_synthesize

    examples = [Example({"x": x}, x + 1) for x in range(3)]
    rng = np.random.default_rng(0)
    best, _trace, verified = public_synthesize(
        ["x"], examples, population_size=30, max_generations=10, max_depth=3,
        allow_motif_bias=True, allow_fold_template=True, allow_recursion_template=True, rng=rng,
    )
    assert best is not None
    assert isinstance(verified, bool)


def test_motif_bias_does_not_yet_solve_leap_year_without_bool_template():
    """Honest measured result for `docs/ROADMAP.md` v0.28. Motif resonance
    (`allow_motif_bias`, see `motif_bias.py`) was built as a general
    alternative to hand-designing a new structural template per target - the
    concern that v0.14/v0.24/v0.27's pattern (notice a failure shape, hand-
    design a skeleton with named holes) doesn't scale to every future
    problem shape, raised directly against this project's own stated
    preference for rules that emerge from the dynamics over special-cased
    symbolic paths (`CLAUDE.md`). Measured, not assumed, against the same
    three targets that motivated each existing template, with that
    template disabled:

    - Leap year (`allow_bool_template=False`, `allow_motif_bias=True`, same
      budget as `test_leap_year_rule_is_reliable_across_seeds`): **0/10
      seeds verified** - every seed converges to the identical
      `not (year % 2)` "is even" trap that plain blind growth hits with no
      template at all (see `_bool_template`'s own module comment). This is
      the representative case asserted below.
    - `sum_of_squares_via_fold` (`allow_fold_template=False`,
      `allow_motif_bias=True`, same budget as
      `test_synthesize_recovers_sum_of_squares_via_fold`): **1/8** - worse
      than blind growth's own historical 3/8 baseline at this exact budget
      (v0.22's audit), though one seed (7) did rediscover the genuine,
      generalizing fold shape from scratch via motif reuse in 4 generations
      - a real signal the mechanism *can* work, not a total failure.
    - `2**n` (`allow_recursion=True`, `allow_recursion_template=False`,
      `allow_motif_bias=True`, same budget as
      `test_resonant_bias_discovers_recursion_reliably_across_seeds`):
      **1/8 "verified", 0/8 generalizing** - the one verified seed found a
      coincidental non-recursive expression that merely fits the 5 training
      examples, the exact verified-but-wrong failure shape this project's
      own methodology exists to catch, not genuine recursion discovery.

    Diagnosis: motif resonance can only reinforce and reuse structure that
    has already appeared *somewhere* in the population with a competitive
    energy - it has no way to independently invent a compound shape that
    essentially never spontaneously forms under blind growth in the first
    place, which is exactly why each of the three hand-built templates was
    needed (`_bool_template`'s own comment: blind growth reliably misses the
    3-atom AND/OR nesting; `_recursive_template`'s docstring: "under 5% of
    random depth-4 trees even contain a `Letrec` with an `If`-shaped body").
    The very first generation is 100% blind growth, so if the correct
    top-level shape never appears there, the archive has nothing genuine to
    discover before the population commits to a coincidental local optimum
    instead - and once it does, motif resonance reinforces *that*
    structure's pieces just as readily as it would reinforce a correct one.
    `sum_of_squares_via_fold`'s partial, real success fits this diagnosis
    exactly: a `Fold` node itself is common under blind growth (unlike a
    3-atom boolean formula or a `Letrec`), so the hard part there is a
    comparatively small "which transform" choice inside an already-common
    skeleton - closer to what a subtree-reuse mechanism is actually suited
    for.

    This is not a claim the general mechanism is a dead end - it ships as a
    genuine, fully generic, zero-regression opt-in (see
    `test_motif_bias_is_a_true_noop_by_default` above) precisely so future
    work can build on it - but it is not, today, a replacement for
    `_recursive_template`/`_fold_template`/`_bool_template`, and this
    project's own practice is to record that honestly rather than claim
    success it hasn't measured.
    """
    leap_years = [
        2023, 2021, 2019, 1999, 2001, 2024, 1996, 2004, 1988, 2012, 2020, 1980,
        1900, 1800, 1700, 2100, 2200, 2300, 2000, 1600, 2400,
    ]
    examples = [Example({"year": y}, y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) for y in leap_years]
    rng = np.random.default_rng(0)
    best, trace, verified = synthesize(
        ["year"], examples, population_size=300, max_generations=150, max_depth=4,
        allow_bool_template=False, allow_motif_bias=True, rng=rng,
    )
    assert not verified
    assert len(trace) == 150


def test_resonant_bias_has_evidence():
    """`has_evidence` (added for `grammar_bias.py` - see `docs/ROADMAP.md`
    v0.29) is the public equivalent of checking `_accum` from outside the
    class, needed so callers can fall back to their own default draw rather
    than sampling a still-uniform, no-evidence distribution."""
    cb = Codebook(dim=64, seed=0)
    rb = ResonantBias(cb, {"op": ["+", "-"]})
    assert not rb.has_evidence("op")
    rb.reinforce({"op": "+"}, weight=1.0)
    assert rb.has_evidence("op")
    assert not rb.has_evidence("nonexistent")


def test_grammar_bias_registers_and_samples():
    """Unit-level check of `GrammarBias` in isolation (see
    `grammar_bias.py`), independent of whether the mechanism helps any
    particular search target (see the measured results below)."""
    cb = Codebook(dim=128, seed=0)
    gb = GrammarBias(ResonantBias(cb, {}), GRAMMAR_CHOICE_VOCAB)
    ctx_choice = "binop_op"
    fallback_calls = []

    def fallback():
        fallback_calls.append(1)
        return "FALLBACK"

    # No evidence yet -> falls back.
    result = gb.sample_or(ctx_choice, "root", 0, np.random.default_rng(0), fallback)
    assert result == "FALLBACK"
    assert len(fallback_calls) == 1

    tree = BinOp("and", Const(1), Const(2))
    gb.reinforce_from_population([tree], np.array([0.1]))
    ctx = production_context("binop_op", "root", 0)
    assert ctx in gb.bias.categories
    assert gb.bias.categories[ctx] == GRAMMAR_CHOICE_VOCAB["binop_op"]
    assert gb.bias.has_evidence(ctx)

    # Now has evidence -> resonance-samples instead of falling back.
    result = gb.sample_or(ctx_choice, "root", 0, np.random.default_rng(0), fallback)
    assert result in GRAMMAR_CHOICE_VOCAB["binop_op"]
    assert len(fallback_calls) == 1  # unchanged - fallback not called again


def test_collect_production_choices_yields_contexts_matching_grow_sites():
    """The `(parent_kind, child_slot)` convention must match what `_grow`
    itself threads through its own recursion (added in v0.28, reused here) -
    otherwise a choice observed here would never be addressable at the
    grow-site that could actually reuse it."""
    tree = BinOp("and", UnaryOp("not", Var("x")), Const(3))
    found = {(ctx, name): value for ctx, name, value in collect_production_choices(tree)}
    assert found[(production_context("binop_op", "root", 0), "binop_op")] == "and"
    assert found[(production_context("unaryop_op", "BinOp", 0), "unaryop_op")] == "not"
    assert found[(production_context("leaf_kind", "UnaryOp", 0), "leaf_kind")] == "var"
    assert found[(production_context("leaf_kind", "BinOp", 1), "leaf_kind")] == "const"
    assert found[(production_context("leaf_const", "BinOp", 1), "leaf_const")] == 3


def test_grammar_bias_save_and_load_round_trip(tmp_path):
    """Persistence is the point of `grammar_bias.py` beyond what
    `motif_bias.py` already offers - `save_grammar_bias`/`load_grammar_bias`
    must actually restore sampling behavior, not just deserialize without
    error. `Codebook.symbol`'s order-dependence (see
    `test_codebook_items_and_load_round_trip` in `test_hypervectors.py`)
    means this only works if the codebook's minted vectors are persisted
    directly - checked here end-to-end, not just at the `Codebook` unit
    level."""
    cb = Codebook(dim=128, seed=0)
    gb = GrammarBias(ResonantBias(cb, {}), GRAMMAR_CHOICE_VOCAB)
    tree = BinOp("%", Var("x"), Const(4))
    gb.reinforce_from_population([tree], np.array([0.05]))
    ctx = production_context("binop_op", "root", 0)
    assert gb.bias.has_evidence(ctx)

    path = str(tmp_path / "grammar_bias.pkl")
    save_grammar_bias(gb, path)
    loaded = load_grammar_bias(path, GRAMMAR_CHOICE_VOCAB)

    assert loaded.bias.has_evidence(ctx)
    assert loaded.bias.categories[ctx] == gb.bias.categories[ctx]
    # Same accumulated evidence -> same sampling distribution, not just "no
    # crash on load": both should agree, given the identical rng draw.
    original_sample = gb.sample_or("binop_op", "root", 0, np.random.default_rng(1), lambda: None)
    loaded_sample = loaded.sample_or("binop_op", "root", 0, np.random.default_rng(1), lambda: None)
    assert original_sample == loaded_sample


def test_grammar_bias_is_a_true_noop_by_default():
    """`allow_grammar_bias=False` (the default) must leave every existing
    caller's behavior bit-for-bit unchanged, mirroring
    `test_motif_bias_is_a_true_noop_by_default`. The new draws added to
    `_grow`'s `binop`/`unaryop` branches and `_leaf` are guarded behind
    `if grammar_bias is not None`, and every entry point (`random_program`,
    `synthesize`) only ever forwards a real `GrammarBias` into `_grow` when
    `allow_grammar_bias=True` - so passing `allow_grammar_bias=False` must
    produce identical output to not mentioning the parameter at all, even
    when a `grammar_bias` object is (harmlessly) also supplied."""
    cb = Codebook(dim=64, seed=0)
    gb = GrammarBias(ResonantBias(cb, {}), GRAMMAR_CHOICE_VOCAB)
    gb.reinforce_from_population([BinOp("and", Const(1), Const(2))], np.array([0.01]))

    program_a = random_program(["x"], np.random.default_rng(7), max_depth=4)
    program_b = random_program(
        ["x"], np.random.default_rng(7), max_depth=4, allow_grammar_bias=False, grammar_bias=gb
    )
    assert pretty(program_a) == pretty(program_b)

    examples = [Example({"x": x}, x + 1) for x in range(4)]
    best_c, trace_c, verified_c = synthesize(
        ["x"], examples, population_size=30, max_generations=10, max_depth=3, rng=np.random.default_rng(2)
    )
    best_d, trace_d, verified_d = synthesize(
        ["x"], examples, population_size=30, max_generations=10, max_depth=3,
        allow_grammar_bias=False, grammar_bias=gb, rng=np.random.default_rng(2),
    )
    assert pretty(best_c) == pretty(best_d)
    assert trace_c == trace_d
    assert verified_c == verified_d


def test_synth_wrapper_exposes_allow_grammar_bias():
    """Mirrors `test_synth_wrapper_exposes_allow_bool_template`/
    `test_synth_wrapper_exposes_allow_motif_bias`: guards against the same
    class of bug recurring for the parameters this entry added."""
    from zeuss.tier4_synthesis.synth import synthesize as public_synthesize

    examples = [Example({"x": x}, x + 1) for x in range(3)]
    rng = np.random.default_rng(0)
    best, _trace, verified = public_synthesize(
        ["x"], examples, population_size=30, max_generations=10, max_depth=3,
        allow_grammar_bias=True, grammar_bias=None, rng=rng,
    )
    assert best is not None
    assert isinstance(verified, bool)


def test_grammar_bias_persists_across_synthesize_calls():
    """The one capability motif resonance doesn't have: a caller-supplied
    `GrammarBias` is reinforced in place across multiple calls to
    `synthesize`, rather than being rebuilt and discarded every call (see
    `synthesize`'s docstring on the ownership rule this implies)."""
    cb = Codebook(dim=128, seed=1)
    persistent = GrammarBias(ResonantBias(cb, {}), GRAMMAR_CHOICE_VOCAB)
    # x*7 (used elsewhere in this file as a "deliberately hard" target) runs
    # its full generation budget without verifying at this tiny population,
    # unlike x+1 - guaranteeing several generations of reinforcement happen
    # before either call returns, rather than risking a generation-0 solve
    # that breaks out of the loop before any reinforcement block runs.
    examples = [Example({"x": x}, x * 7) for x in range(1, 4)]

    synthesize(
        ["x"], examples, population_size=20, max_generations=15, max_depth=2,
        allow_grammar_bias=True, grammar_bias=persistent, rng=np.random.default_rng(2),
    )
    assert any(persistent.bias.categories.values())  # first call left evidence behind

    evidence_before = {ctx: persistent.bias.has_evidence(ctx) for ctx in persistent.bias.categories}

    synthesize(
        ["x"], examples, population_size=20, max_generations=15, max_depth=2,
        allow_grammar_bias=True, grammar_bias=persistent, rng=np.random.default_rng(3),
    )
    # The object passed in is still the one accumulating - not silently
    # replaced by an internally-constructed fresh one (the "owned" branch
    # only applies when the caller passes grammar_bias=None).
    assert all(persistent.bias.has_evidence(ctx) for ctx in evidence_before)


def test_grammar_bias_improves_sum_of_squares_via_fold_over_baseline():
    """Honest measured positive result for `docs/ROADMAP.md` v0.29,
    complementing the negative result below. Production-level PCFG
    resonance (`allow_grammar_bias`, see `grammar_bias.py`) was measured
    against the same three targets v0.28's motif resonance was measured
    against, same budgets/seeds, with the corresponding hand template
    disabled:

    - `sum_of_squares_via_fold` (`allow_fold_template=False`, budget/seeds
      from `test_synthesize_recovers_sum_of_squares_via_fold`, 8 seeds):
      **4/8 verified and generalizing** - genuinely better than blind
      growth's own historical 3/8 baseline at this exact budget (v0.22's
      audit) *and* better than motif resonance's 1/8 (v0.28) on the same
      target. This is real, measured lift, not assumed: a `Fold` node is
      already common under blind growth (weight 3 in
      `_KIND_WEIGHTS_WITH_LIST`), so the hard part here is a comparatively
      small "which transform" choice - exactly the shape of problem
      per-choice marginal reinforcement is suited for, unlike a rare
      top-level compound shape (see the negative result below).
    - Leap year: **0/10** - identical `not (year % 2)` trap to motif
      resonance's own result. Diagnosed why, not just reported: blind
      growth converges to this local optimum *fast*, so
      `reinforce_from_population` ends up reinforcing that wrong
      structure's own production choices (the `%`/`not`/`2` pattern) more
      than anything else once it dominates the population - the bias
      actively pushes *harder* toward reproducing the wrong answer instead
      of escaping it. This is the same premature-lock-in failure mode
      motif resonance hit, arrived at by a different route - see the
      dedicated negative-result test below.
    - `2**n`: **0/8 verified** (worse than motif resonance's 1/8-but-
      non-generalizing) - `Letrec`/`Recur` essentially never appears under
      blind growth in the first place (`_recursive_template`'s own
      docstring: under 5% of random depth-4 trees), so there is very
      little genuine recursion-shaped evidence to ever reinforce; the
      population instead converges on non-recursive arithmetic
      coincidences (`n*n`, `if n then n else 1`) and the bias reinforces
      *those* instead - the same lock-in shape as leap year, wearing a
      different hat.

    Net finding: motif resonance and grammar resonance are not simply
    "the second one is strictly better" - each helps on a *different*
    target shape (grammar resonance materially helps the fold target motif
    resonance didn't; neither helps the two targets whose difficulty is a
    rare top-level construct essentially never appearing under blind growth
    at all) and both share the same root failure mode once a wrong
    structure dominates a population early. Neither replaces
    `_recursive_template`/`_fold_template`/`_bool_template` - this is the
    representative *positive* case, held to the same standard as every
    other reliability claim in this file: verified *and* generalizing,
    seeds picked from the full 8-seed sweep that were measured to succeed.
    """
    examples = [
        Example({"xs": [1, 2, 3]}, 14),
        Example({"xs": [4, 5]}, 41),
        Example({"xs": [10]}, 100),
        Example({"xs": []}, 0),
        Example({"xs": [-1, -2]}, 5),
        Example({"xs": [0, 0, 0]}, 0),
    ]
    held_out = ([1, 1, 1, 1], [3, -3], [], [7], [2, 2, 2], [6, -1, 4])
    for seed in (4, 7):
        rng = np.random.default_rng(seed)
        best, _trace, verified = synthesize(
            ["xs"], examples, list_inputs=("xs",), population_size=300, max_generations=100, max_depth=3,
            allow_fold_template=False, allow_grammar_bias=True, rng=rng,
        )
        assert verified, f"seed {seed} failed to verify"
        for xs in held_out:
            assert evaluate(best, {"xs": xs}, Fuel(500)) == sum(x * x for x in xs), f"seed {seed} didn't generalize"


def test_grammar_bias_does_not_yet_solve_leap_year_or_pow2():
    """The negative half of the honest v0.29 result (see the docstring of
    `test_grammar_bias_improves_sum_of_squares_via_fold_over_baseline`
    above for the full three-target measurement and diagnosis). Asserted at
    one representative seed each, mirroring
    `test_motif_bias_does_not_yet_solve_leap_year_without_bool_template`'s
    pattern rather than re-running the full sweep in a committed test."""
    leap_years = [
        2023, 2021, 2019, 1999, 2001, 2024, 1996, 2004, 1988, 2012, 2020, 1980,
        1900, 1800, 1700, 2100, 2200, 2300, 2000, 1600, 2400,
    ]
    leap_examples = [Example({"year": y}, y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) for y in leap_years]
    rng = np.random.default_rng(0)
    best, trace, verified = synthesize(
        ["year"], leap_examples, population_size=300, max_generations=150, max_depth=4,
        allow_bool_template=False, allow_grammar_bias=True, rng=rng,
    )
    assert not verified
    assert len(trace) == 150

    pow2_examples = [Example({"n": n}, 2**n) for n in range(5)]
    rng2 = np.random.default_rng(0)
    best2, _trace2, verified2 = synthesize(
        ["n"], pow2_examples, population_size=800, max_generations=150, max_depth=4,
        allow_recursion=True, allow_recursion_template=False, allow_grammar_bias=True,
        fuel_budget=200, rng=rng2,
    )
    assert not verified2


def test_root_shape_signature():
    """`root_shape` (see `search.py`, built for `allow_shape_elitism` -
    `docs/ROADMAP.md` v0.30) is deliberately hand-declaration-free: just the
    root node's DSL type name plus its `op` attribute when it has one (`None`
    otherwise), unlike `combine_kind`/`base_kind`/`shape_kind`, which only
    exist for the three hand-built templates' own shapes."""
    assert root_shape(BinOp("+", Const(1), Const(2))) == ("BinOp", "+")
    assert root_shape(UnaryOp("not", Const(True))) == ("UnaryOp", "not")
    assert root_shape(Const(3)) == ("Const", None)
    assert root_shape(Var("x")) == ("Var", None)
    assert root_shape(Letrec("f", ("p",), Const(0), Const(1))) == ("Letrec", None)
    # Two BinOps with different operators are different shapes; same operator
    # is the same shape regardless of operands - the whole point is that this
    # groups by structural family, not by exact tree identity.
    assert root_shape(BinOp("+", Const(1), Const(2))) != root_shape(BinOp("-", Const(1), Const(2)))
    assert root_shape(BinOp("+", Const(1), Const(2))) == root_shape(BinOp("+", Var("x"), Var("y")))


def test_shape_elitism_is_a_true_noop_by_default():
    """`allow_shape_elitism=False` (the default) must leave every existing
    caller's behavior bit-for-bit unchanged, mirroring
    `test_motif_bias_is_a_true_noop_by_default`/
    `test_grammar_bias_is_a_true_noop_by_default`. Unlike those two, shape
    elitism never touches `_grow`/`mutate`/`random_program` at all - it only
    reads `population`/`raw_energies` inside `synthesize`'s own generational
    loop (see `root_shape`'s dict there) - so the whole mechanism is gated by
    one `if allow_shape_elitism:` block with no rng draws of its own. Checked
    directly anyway, not just inferred from the code shape, per this
    project's own established practice of confirming true-no-op claims."""
    examples = [Example({"x": x}, x + 1) for x in range(4)]
    best_a, trace_a, verified_a = synthesize(
        ["x"], examples, population_size=30, max_generations=10, max_depth=3, rng=np.random.default_rng(2)
    )
    best_b, trace_b, verified_b = synthesize(
        ["x"], examples, population_size=30, max_generations=10, max_depth=3,
        allow_shape_elitism=False, rng=np.random.default_rng(2),
    )
    assert pretty(best_a) == pretty(best_b)
    assert trace_a == trace_b
    assert verified_a == verified_b


def test_synth_wrapper_exposes_allow_shape_elitism():
    """Mirrors `test_synth_wrapper_exposes_allow_grammar_bias`: guards
    against the same class of bug recurring for the parameter this entry
    added."""
    from zeuss.tier4_synthesis.synth import synthesize as public_synthesize

    examples = [Example({"x": x}, x + 1) for x in range(3)]
    rng = np.random.default_rng(0)
    best, _trace, verified = public_synthesize(
        ["x"], examples, population_size=30, max_generations=10, max_depth=3,
        allow_shape_elitism=True, rng=rng,
    )
    assert best is not None
    assert isinstance(verified, bool)


def test_and_or_not_target_decomposition():
    """Direct truth-table checks for `semantic_bias.py`'s decomposition
    functions - independent of any search run. `and_left_target`/
    `or_left_target` only commit a hard constraint on the side that's
    provably required regardless of the other child (AND needs both `True`
    to reach `True`; OR needs both `False` to reach `False`); everywhere
    else is `None` (don't-care) since the *other* child could still satisfy
    the row. `and_right_target`/`or_right_target` need a concrete
    `left_actual` (unlike the left versions) precisely because `AND(True,
    r)=r`/`OR(False, r)=r` only resolve once `l` is actually known."""
    assert and_left_target([True, False, None]) == [True, None, None]
    assert and_right_target([True, True, False], [True, False, True]) == [True, None, False]
    assert or_left_target([True, False, None]) == [None, False, None]
    assert or_right_target([True, False, False], [False, True, False]) == [True, None, False]
    assert not_target([True, False, None]) == [False, True, None]


def test_best_atom_for_target_finds_exact_match():
    """`best_atom_for_target` (see `semantic_bias.py`) must find an atom that
    exactly reproduces a known target vector when one exists in
    `BOOL_TEMPLATE_CATEGORIES`'s vocabulary - the vectorized numpy scoring
    is the one genuinely new piece of logic this module adds over the
    existing templates' plain uniform/resonance draws, so it gets a direct
    unit test rather than relying only on the end-to-end target test below."""
    years = [2023, 2024, 2000, 1900, 1996, 2100]
    examples = [Example({"year": y}, None) for y in years]
    targets = [(y % 4 == 0) for y in years]
    node = best_atom_for_target(
        ["year"], examples, targets, BOOL_TEMPLATE_CATEGORIES, _build_atom_node, np.random.default_rng(0)
    )
    for ex, t in zip(examples, targets):
        assert evaluate(node, dict(ex.inputs), Fuel(50)) == t

    # Don't-care rows are excluded from scoring entirely, not treated as
    # False - a target that's only constrained on odd-indexed rows must
    # still be matched by an atom scored only against those rows.
    partial_targets = [True, None, False, None, True, None]
    node2 = best_atom_for_target(
        ["year"], examples, partial_targets, BOOL_TEMPLATE_CATEGORIES, _build_atom_node, np.random.default_rng(0)
    )
    for ex, t in zip(examples, partial_targets):
        if t is not None:
            assert evaluate(node2, dict(ex.inputs), Fuel(50)) == t


def test_best_atom_for_target_all_dont_care_still_returns_a_node():
    """When every row is don't-care (e.g. an AND/OR ancestor has already
    proven this whole subtree's value can't affect correctness), there is
    nothing to score against - but the caller still needs a real node back,
    not `None`, or the entire candidate would be discarded over a position
    nothing depends on. A uniformly random atom is returned instead."""
    examples = [Example({"year": y}, None) for y in (2000, 2001, 2002)]
    node = best_atom_for_target(
        ["year"], examples, [None, None, None], BOOL_TEMPLATE_CATEGORIES, _build_atom_node, np.random.default_rng(0)
    )
    assert node is not None
    evaluate(node, dict(examples[0].inputs), Fuel(50))  # must not raise

    assert best_atom_for_target([], examples, [None], BOOL_TEMPLATE_CATEGORIES, _build_atom_node, np.random.default_rng(0)) is None


def test_grow_boolean_targeted_reconstructs_a_known_and_or3_formula():
    """Round-trip check of `grow_boolean_targeted`/`boolean_backprop_template`
    independent of the full GP search: given examples generated from a
    *known* and_or3-shaped formula, the returned node must evaluate
    correctly against every one of them - confirming the AND/OR
    decomposition and the atom search compose correctly end to end, not
    just in isolation."""
    def true_rule(year):
        return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

    years = [
        2023, 2021, 2019, 1999, 2001, 2024, 1996, 2004, 1988, 2012, 2020, 1980,
        1900, 1800, 1700, 2100, 2200, 2300, 2000, 1600, 2400,
    ]
    examples = [Example({"year": y}, true_rule(y)) for y in years]

    # A single blind draw isn't guaranteed to land on a fully-matching
    # decomposition (the "kind" draw at each node is uniform/fixed, exactly
    # to avoid premature structural lock-in - see _KIND_WEIGHTS's comment),
    # so this tries a bounded number of seeds and asserts at least one
    # succeeds - proving the mechanism *can* reconstruct the known shape
    # end to end (measured directly: roughly 1 in 150 single, non-iterated
    # draws does, at this exact example set - see docs/ROADMAP.md v0.31),
    # not that every single draw does.
    for seed in range(500):
        node = boolean_backprop_template(
            ["year"], (), examples, np.random.default_rng(seed), 4, BOOL_TEMPLATE_CATEGORIES, _build_atom_node, 60
        )
        if node is not None and all(
            evaluate(node, dict(ex.inputs), Fuel(50)) == ex.expected_output for ex in examples
        ):
            break
    else:
        raise AssertionError("no seed in range(500) reconstructed a fully-matching formula")


def test_boolean_backprop_template_returns_none_for_non_boolean_targets():
    """The mechanism is deliberately scoped to boolean compound targets only
    (see `semantic_bias.py`'s module docstring) - it must not try to apply
    to an arithmetic target, mirroring `_bool_template`'s own `(None, None)`
    "nothing to offer" contract for its own inapplicable case (no scalar
    input)."""
    examples = [Example({"x": x}, x + 1) for x in range(4)]
    node = boolean_backprop_template(
        ["x"], (), examples, np.random.default_rng(0), 4, BOOL_TEMPLATE_CATEGORIES, _build_atom_node, 60
    )
    assert node is None


def test_semantic_bias_is_a_true_noop_by_default():
    """`allow_semantic_bias=False` (the default) must leave every existing
    caller's behavior bit-for-bit unchanged, mirroring
    `test_grammar_bias_is_a_true_noop_by_default`/
    `test_shape_elitism_is_a_true_noop_by_default`. Checked directly, not
    just inferred from the ``if allow_semantic_bias and examples`` guard's
    shape, per this project's own established practice for true-no-op
    claims."""
    examples = [Example({"x": x}, x + 1) for x in range(4)]
    program_a = random_program(["x"], np.random.default_rng(7), max_depth=4)
    program_b = random_program(
        ["x"], np.random.default_rng(7), max_depth=4, allow_semantic_bias=False, examples=examples
    )
    assert pretty(program_a) == pretty(program_b)

    best_c, trace_c, verified_c = synthesize(
        ["x"], examples, population_size=30, max_generations=10, max_depth=3, rng=np.random.default_rng(2)
    )
    best_d, trace_d, verified_d = synthesize(
        ["x"], examples, population_size=30, max_generations=10, max_depth=3,
        allow_semantic_bias=False, rng=np.random.default_rng(2),
    )
    assert pretty(best_c) == pretty(best_d)
    assert trace_c == trace_d
    assert verified_c == verified_d


def test_synth_wrapper_exposes_allow_semantic_bias():
    """Mirrors `test_synth_wrapper_exposes_allow_shape_elitism`: guards
    against the same class of bug recurring for the parameter this entry
    added."""
    from zeuss.tier4_synthesis.synth import synthesize as public_synthesize

    leap_years = [2023, 2021, 2019, 1999, 2001, 2024, 1996, 2004]
    examples = [Example({"year": y}, y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) for y in leap_years]
    rng = np.random.default_rng(0)
    best, _trace, verified = public_synthesize(
        ["year"], examples, population_size=30, max_generations=10, max_depth=4,
        allow_bool_template=False, allow_semantic_bias=True, rng=rng,
    )
    assert best is not None
    assert isinstance(verified, bool)


def test_semantic_bias_solves_leap_year_reliably_across_seeds():
    """Honest measured positive result for `docs/ROADMAP.md` v0.31 - the
    actual point of this whole mechanism. Six prior general mechanisms
    (three hand-built templates aside) all failed leap year identically:
    motif resonance 0/10 (v0.28), grammar resonance 0/10 (v0.29), shape
    elitism alone/+motif/+grammar 0/10 each (v0.30) - every one of them only
    reinforces/protects/reshuffles individuals blind growth already
    produced, powerless if the correct compound shape essentially never
    gets generated at all. Semantic backpropagation acts at generation time
    instead, and at this exact configuration
    (`test_leap_year_rule_is_reliable_across_seeds`'s own budget/seeds,
    `allow_bool_template=False` so `_bool_template` can't be doing the
    work): **30/30 seeds verified** (measured directly, including seed 19 -
    a genuine residual failure recorded against `_bool_template` itself in
    `docs/ROADMAP.md`'s v0.20-era audit) and **27/30 verified and
    generalizing** to the held-out years below. Seeds 0-7 (this test's own
    range, mirroring the existing test's convention) are 8/8 verified and
    generalizing - the representative subset asserted here, not the full
    sweep, for the same runtime reason every other reliability test in this
    file only checks a subset.

    The 3/30 non-generalizing seeds are an honest, expected caveat, not
    swept under the rug: e.g. seed 27 finds ``(year % 2 == 0) and (not
    (year % 100 <= 6) or year % 400 < 5)`` - a coincidental century clause
    (``year % 100 <= 6`` instead of ``year % 100 == 0``) that happens to
    match every training century-year but misclassifies the held-out 1904
    - the same "verified means matched the given examples, never proven
    correct" honesty this module's own docstring (`synth.py`) already
    states, and the same character of caveat `test_resonant_bias_can_
    discover_fibonacci`'s seed-7 case already has on record.
    """
    leap_years = [
        2023, 2021, 2019, 1999, 2001,
        2024, 1996, 2004, 1988, 2012, 2020, 1980,
        1900, 1800, 1700, 2100, 2200, 2300,
        2000, 1600, 2400,
    ]
    examples = [Example({"year": y}, y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) for y in leap_years]
    held_out = [
        (2025, False), (2028, True), (1904, True), (1596, True),
        (1960, True), (2008, True), (1500, False), (3000, False), (2044, True),
    ]
    for seed in range(8):
        rng = np.random.default_rng(seed)
        best, _trace, verified = synthesize(
            ["year"], examples, population_size=300, max_generations=150, max_depth=4,
            allow_bool_template=False, allow_semantic_bias=True, rng=rng,
        )
        assert verified, f"leap year seed {seed} found nothing"
        for year, expected in held_out:
            assert evaluate(best, {"year": year}, Fuel(200)) == expected, f"seed {seed} didn't generalize"
