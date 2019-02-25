"""JavaScript d3 static and dynamic visualization layouts."""


import pandas as pd

from string import Template
from textwrap import dedent
from pathlib import Path

from IPython.core.display import display, Javascript

from thoth.lab.jupyter import load_style


_THIS_DIR = Path(__file__).parent

_DEFAULT_CSS_DIR = _THIS_DIR.parent / Path("assets/css")

_LIB_D3 = 'https://d3js.org/d3.v5.min'
_LIB_D3_HIERARCHY = 'https://d3js.org/d3-hierarchy.v1.min'

_REQUIREJS_TEMPLATE = Template(dedent("""
    require.config({
        paths: {
            $libs
        }
    });
"""))


def init_notebook_mode(custom_css: list = None, custom_libs: dict = None):
    """Initialize notebook mode by linking required d3 libraries.

    :param custom_css: list of custom css urls to link

        NOTE: It is not possible to link locally located css files.

    :param custom_libs: custom JavaScript libraries to link

        The libraries are linked using requirejs such as:
        ```python
        require.config({ paths: {<key>: <path>} });
        ```

        Please note that <path> does __NOT__ contain `.js` suffix.
    """
    custom_libs = custom_libs or {}

    # required styles
    style_css = '\n'.join([
        f"{css_file.read_text()}"
        for css_file in _DEFAULT_CSS_DIR.iterdir()
    ])

    load_style(style_css)

    # required libraries
    require = (
        f"'d3': '{_LIB_D3}'",
        f"'d3-hierarchy': '{_LIB_D3_HIERARCHY}'",
        *(f"'{key}': '{path}'" for key, path in custom_libs.items())
    )
    require_js: str = _REQUIREJS_TEMPLATE.safe_substitute(libs=', '.join(require))

    return display(Javascript(dedent(require_js), css=custom_css))


def plot(data: pd.DataFrame,
         kind: str = 'diagonal',
         **kwargs):
    """Syntactic sugar which wraps static plot visualizations."""
    template: Template = _get_js_template(kind, static=True)

    js: str = template.safe_substitute(
        data=data.to_csv(index=False),
        **kwargs
    )

    return display(Javascript(data=js))


def iplot(data: pd.DataFrame,
          kind: str = 'diagonal',
          **kwargs):
    """Syntactic sugar which wraps dynamic plot visualizations."""
    template: Template = _get_js_template(kind, static=False)

    js: str = template.safe_substitute(
        data=data.to_csv(index=False),
        **kwargs
    )

    return display(Javascript(data=js))


def _get_js_template(kind: str, static: bool = True) -> Template:
    """Return string template of JS script."""
    script_path = _THIS_DIR / Path(
        f"{['dynamic', 'static'][static]}/templates/{kind}.js")

    return Template(script_path.read_text())
