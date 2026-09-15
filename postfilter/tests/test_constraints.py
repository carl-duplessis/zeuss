from postfilter.constraints import disjoint_category_exclusions, no_cycle_exclusions


def test_no_cycle_exclusions_walks_full_subtree():
    # x -> a -> b  (children[x] = {a}, children[a] = {b})
    children = {"x": {"a"}, "a": {"b"}}
    assert no_cycle_exclusions(children, "x") == {"a", "b"}


def test_no_cycle_exclusions_empty_for_a_leaf():
    children = {"x": {"a"}}
    assert no_cycle_exclusions(children, "a") == set()


def test_no_cycle_exclusions_unknown_subject_is_empty():
    assert no_cycle_exclusions({"x": {"a"}}, "nowhere") == set()


def test_disjoint_category_exclusions_basic():
    category_of = {"tree": "plant", "person1": "person", "rock": "mineral"}
    groups = [{"plant", "person", "animal"}]
    excluded = disjoint_category_exclusions(
        category_of, groups, "person1", ["tree", "rock", "person1"]
    )
    # "tree" is in the disjoint group and a different category -> excluded.
    # "rock" (mineral) is outside the disjoint group entirely -> not excluded.
    # "person1" is the subject itself and shares its own category -> not excluded.
    assert excluded == {"tree"}


def test_disjoint_category_exclusions_no_group_membership_is_a_noop():
    category_of = {"a": "widget", "b": "gadget"}
    groups = [{"plant", "person"}]
    assert disjoint_category_exclusions(category_of, groups, "a", ["b"]) == set()


def test_disjoint_category_exclusions_unknown_subject_is_empty():
    category_of = {"a": "widget"}
    groups = [{"widget", "gadget"}]
    assert disjoint_category_exclusions(category_of, groups, "nowhere", ["a"]) == set()
