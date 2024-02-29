import glob
import os
import shutil
import invoke
from invoke import Collection

from compas_invocations import build
from compas_invocations import docs as docs_
from compas_invocations import tests
from compas_invocations.console import chdir


@invoke.task(
    help={
        "docs": "True to clean up generated documentation, otherwise False",
        "bytecode": "True to clean up compiled python files, otherwise False.",
        "builds": "True to clean up build/packaging artifacts, otherwise False.",
    }
)
def clean(ctx, docs=True, bytecode=True, builds=True, ghuser=True):
    """Cleans the local copy from compiled artifacts."""

    with chdir(ctx.base_folder):
        if bytecode:
            for root, dirs, files in os.walk(ctx.base_folder):
                for f in files:
                    if f.endswith(".pyc"):
                        os.remove(os.path.join(root, f))
                if ".git" in dirs:
                    dirs.remove(".git")

        folders = []

        if docs:
            folders.append("docs/api/generated")

        folders.append("dist/")

        if bytecode:
            for t in ("src", "tests"):
                folders.extend(glob.glob("{}/**/__pycache__".format(t), recursive=True))

        if builds:
            folders.append("build/")
            folders.extend(glob.glob("src/**/*.egg-info", recursive=False))

        if ghuser and ctx.get("ghuser"):
            folders.append(os.path.abspath(ctx.ghuser.target_dir))

        for folder in folders:
            shutil.rmtree(os.path.join(ctx.base_folder, folder), ignore_errors=True)


@invoke.task()
def lint(ctx):
    """Check the consistency of coding style."""

    print("\nRunning ruff linter...")
    ctx.run("ruff check --fix src tests")

    print("\nRunning black linter...")
    ctx.run("black --check --diff --color src tests")

    print("\nAll linting is done!")


@invoke.task()
def format(ctx):
    """Reformat the code base using black."""

    print("\nRunning ruff formatter...")
    ctx.run("ruff format src tests")

    print("\nRunning black formatter...")
    ctx.run("black src tests")

    print("\nAll formatting is done!")


@invoke.task()
def check(ctx):
    """Check the consistency of documentation, coding style and a few other things."""

    with chdir(ctx.base_folder):
        lint(ctx)


@invoke.task(
    help={
        "rebuild": "True to clean all previously built docs before starting, otherwise False.",
        "doctest": "True to run doctests, otherwise False.",
        "check_links": "True to check all web links in docs for validity, otherwise False.",
    }
)
def docs(ctx, doctest=False, rebuild=False, check_links=False):
    """Builds the HTML documentation."""

    if rebuild:
        clean(ctx)

    with chdir(ctx.base_folder):
        if doctest:
            ctx.run("pytest --doctest-modules")

        opts = "-E" if rebuild else ""
        ctx.run("sphinx-build {} -b html docs dist/docs".format(opts))

        if check_links:
            linkcheck(ctx, rebuild=rebuild)


@invoke.task()
def linkcheck(ctx, rebuild=False):
    """Check links in documentation."""
    print("Running link check...")
    opts = "-E" if rebuild else ""
    ctx.run("sphinx-build {} -b linkcheck docs dist/docs".format(opts))


ns = Collection(
    docs_.help,
    check,
    lint,
    format,
    docs,
    linkcheck,
    tests.test,
    tests.testdocs,
    tests.testcodeblocks,
    build.prepare_changelog,
    clean,
    build.release,
    build.build_ghuser_components,
)
ns.configure(
    {
        "base_folder": os.path.dirname(__file__),
        "ghuser": {
            "source_dir": "src/compas_ghpython/components",
            "target_dir": "src/compas_ghpython/components/ghuser",
            "prefix": "COMPAS: ",
        },
    }
)
