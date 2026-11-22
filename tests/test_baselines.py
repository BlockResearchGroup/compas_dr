"""Comparison of the numerical routines against the recorded COMPAS 2 baselines.

The fixtures in ``tests/baselines`` are read-only (see the README there).
Every case of ``tests/baselines/cases.py`` is re-run; the regenerated inputs
must equal the stored inputs exactly, integer outputs must be equal, and float
outputs must satisfy ``abs(new - base) <= ATOL + RTOL * abs(base)``.

Run with ``-rP`` to see the maximum absolute and relative deviation per
output. The relative deviation is taken over the entries with a non-zero
baseline value.

"""

import importlib.util
import json
from pathlib import Path

import compas
import numpy as np
import pytest

import compas_dr

BASELINES = Path(__file__).parent / "baselines"

ATOL = 1e-9
RTOL = 1e-9


def load_cases():
    spec = importlib.util.spec_from_file_location("compas_dr_baseline_cases", BASELINES / "cases.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cases = load_cases()


def deviation(new, base):
    difference = np.abs(new - base)
    nonzero = base != 0
    absolute = float(difference.max()) if difference.size else 0.0
    relative = float((difference[nonzero] / np.abs(base[nonzero])).max()) if nonzero.any() else 0.0
    return absolute, relative


def test_provenance():
    with open(BASELINES / "provenance.json") as f:
        provenance = json.load(f)

    print("compas {}, compas_dr from {}".format(compas.__version__, compas_dr.__file__))

    assert provenance["package"] == "compas_dr"
    assert provenance["compas"].startswith("2.")
    assert sorted(provenance["cases"]) == sorted(list(cases.CASES) + [cases.CONSTRAINTS_CASE])
    for identifier, status in provenance["cases"].items():
        assert status == "recorded", identifier


@pytest.mark.parametrize("identifier", sorted(cases.CASES))
def test_case(identifier):
    if identifier in cases.REQUIRES:
        pytest.importorskip(cases.REQUIRES[identifier])

    with np.load(BASELINES / (identifier + ".npz"), allow_pickle=False) as fixture:
        stored = {name: fixture[name] for name in fixture.files}

    inputs, outputs = cases.CASES[identifier]()
    computed = {"in_" + name: value for name, value in inputs.items()}
    computed.update({"out_" + name: value for name, value in outputs.items()})

    assert sorted(computed) == sorted(stored)

    case_absolute = 0.0
    case_relative = 0.0
    failures = []

    for name in sorted(stored):
        new = computed[name]
        base = stored[name]

        assert new.shape == base.shape, name
        assert new.dtype == base.dtype, name

        if name.startswith("in_") or base.dtype.kind != "f":
            assert np.array_equal(new, base), name
            continue

        absolute, relative = deviation(new, base)
        case_absolute = max(case_absolute, absolute)
        case_relative = max(case_relative, relative)
        print("{} {}: max abs {:.3e}, max rel {:.3e}".format(identifier, name[4:], absolute, relative))

        if not (np.abs(new - base) <= ATOL + RTOL * np.abs(base)).all():
            failures.append(name)

    if "out_iterations" in stored:
        print("{} iterations: {} (baseline {})".format(identifier, int(computed["out_iterations"]), int(stored["out_iterations"])))
    print("{} CASE: max abs {:.3e}, max rel {:.3e}".format(identifier, case_absolute, case_relative))

    assert not failures


def test_constraints_data():
    """dr-12: the regenerated constraint data equals the recorded COMPAS 2 data."""
    with open(BASELINES / "constraints.json") as f:
        text = f.read()

    stored = compas.json_loads(text)
    computed = cases.constraints()

    assert [type(constraint) for constraint in stored] == [type(constraint) for constraint in computed]
    assert [constraint.__data__ for constraint in stored] == [constraint.__data__ for constraint in computed]
    assert [item["data"] for item in json.loads(text)] == [constraint.__data__ for constraint in computed]
