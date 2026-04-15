# tests/conftest.py
# pytest configuration for Tsugite
#
# The source modules live in setup/ (a flat collection of .py files, not a
# Python package).  pyproject.toml already adds setup/ to sys.path via
# [tool.pytest.ini_options] pythonpath, so imports like `from Misc import …`
# just work.  No additional path manipulation needed here.
#
# Modules that import OpenGL or PyQt5 (Buffer, Show, Types, Geometries, …)
# require a running display/GL context.  Tests for those modules should be
# placed in tests/integration/ and guarded with pytest.mark.skipif or
# pytest-qt fixtures.  The unit tests in this directory only cover modules
# with no GL/Qt dependencies.
