# Sphinx configuration for sphinx-ext-mystmd bridge
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

needs_sphinx = "6.2"
extensions = [
    "myst_parser",
    "sphinx_ext_mystmd",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]
exclude_patterns = ["_build"]

# Napoleon options for numpydoc-style docstrings
napoleon_use_admonition_for_examples = True
napoleon_use_rtype = False
napoleon_use_ivar = True
