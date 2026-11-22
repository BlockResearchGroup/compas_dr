"""Record the COMPAS 2 baselines of ``compas_dr``.

Usage, from the repository root, in the legacy environment::

    <python of brg-compas2> tests/baselines/record.py tests/baselines

The script refuses to run unless

* ``compas`` is a 2.x release,
* ``compas_dr`` is the installed (non-editable) legacy revision, not the
  working tree, and
* the output directory holds no recorded baselines yet.

It writes one ``<case>.npz`` per case of ``cases.py`` (inputs as ``in_*``,
outputs as ``out_*``), ``constraints.json`` and ``provenance.json``. A case
that raises is reported in ``provenance.json`` and gets no fixture.

"""

import datetime
import importlib.metadata
import json
import os
import platform
import sys
import traceback

import numpy as np

LEGACY_REVISION = "84a3b4b3b0017280da73054ac811aab020b43beb"

HERE = os.path.dirname(os.path.abspath(__file__))
REPOSITORY = os.path.dirname(os.path.dirname(HERE))


def check_environment():
    import compas
    import compas_dr

    if not compas.__version__.startswith("2."):
        raise SystemExit("Baselines are recorded with COMPAS 2 only; found compas {}.".format(compas.__version__))

    location = os.path.realpath(compas_dr.__file__)
    if location.startswith(os.path.realpath(REPOSITORY) + os.sep):
        raise SystemExit("compas_dr is imported from the working tree ({}), not from the installed legacy revision.".format(location))

    direct_url = importlib.metadata.distribution("compas_dr").read_text("direct_url.json")
    if not direct_url:
        raise SystemExit("compas_dr has no direct_url.json; its revision cannot be identified.")
    direct_url = json.loads(direct_url)
    if direct_url.get("dir_info", {}).get("editable"):
        raise SystemExit("compas_dr is an editable install; the legacy revision must be installed non-editable.")
    commit = direct_url.get("vcs_info", {}).get("commit_id")
    if commit != LEGACY_REVISION:
        raise SystemExit("compas_dr is installed at {}, not at the legacy revision {}.".format(commit, LEGACY_REVISION))

    return commit


def check_outputs(identifier, outputs):
    for name, value in outputs.items():
        if value.dtype.kind == "f" and not np.isfinite(value).all():
            raise ValueError("Output '{}' of {} is not finite.".format(name, identifier))


def version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main(target):
    commit = check_environment()

    import cases
    import compas

    target = os.path.abspath(target)
    if os.path.exists(os.path.join(target, "provenance.json")):
        raise SystemExit("{} already holds recorded baselines. They are read-only and change only by decision.".format(target))

    status = {}

    for identifier in sorted(cases.CASES):
        try:
            inputs, outputs = cases.CASES[identifier]()
            check_outputs(identifier, outputs)
        except Exception:
            status[identifier] = "not recorded: " + traceback.format_exc().strip().splitlines()[-1]
            print("{}: {}".format(identifier, status[identifier]))
            continue
        arrays = {"in_" + name: value for name, value in inputs.items()}
        arrays.update({"out_" + name: value for name, value in outputs.items()})
        np.savez(os.path.join(target, identifier + ".npz"), **arrays)
        status[identifier] = "recorded"
        print("{}: recorded ({} inputs, {} outputs)".format(identifier, len(inputs), len(outputs)))

    with open(os.path.join(target, "constraints.json"), "w") as f:
        f.write(compas.json_dumps(cases.constraints(), pretty=True) + "\n")
    status[cases.CONSTRAINTS_CASE] = "recorded"
    print("{}: recorded (constraints.json)".format(cases.CONSTRAINTS_CASE))

    provenance = {
        "package": "compas_dr",
        "commit": commit,
        "compas": compas.__version__,
        "compas_occ": version("compas_occ"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": version("scipy"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "environment": os.path.basename(sys.prefix),
        "command": " ".join(["python"] + sys.argv),
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "cases": status,
    }
    with open(os.path.join(target, "provenance.json"), "w") as f:
        json.dump(provenance, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
