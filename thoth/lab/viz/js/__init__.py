"""JavaScript d3 static and dynamic visualization layouts."""


import pandas as pd

from pathlib import Path

from jupyter_require.core import execute_js
from jupyter_require.core import require
from jupyter_require.core import link_css
from jupyter_require.core import load_css


_THIS_DIR = Path(__file__).parent

_DEFAULT_CSS_DIR = _THIS_DIR.parent / Path("assets/css")
_DEFAULT_LIBRARIES = {
    'd3': 'https://d3js.org/d3.v5.min',
    'd3-hierarchy': 'https://d3js.org/d3-hierarchy.v1.min'
}


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
    required_libraries = custom_libs or {}
    required_libraries.update(_DEFAULT_LIBRARIES)

    # required styles
    style_css = '\n'.join([
        f"{css_file.read_text()}"
        for css_file in _DEFAULT_CSS_DIR.iterdir()
    ])

    load_css(style_css, {'id': 'thoth-lab-stylesheet'})

    # custom css links
    for stylesheet in custom_css or []:
        link_css(stylesheet)

    # required libraries
    return require.config(required_libraries)


def plot(data: pd.DataFrame,
         kind: str = 'diagonal',
         layout: str = 'tree',
         **kwargs):
    """Syntactic sugar which wraps static plot visualizations."""
    js: str = _get_js_template(kind, static=True)

    return execute_js(js,
                      data=data.to_csv(index=False),
                      layout=layout,
                      **kwargs)


def iplot(data: pd.DataFrame,
          kind: str = 'diagonal',
          layout: str = 'tree',
          **kwargs):
    """Syntactic sugar which wraps dynamic plot visualizations."""
    js: str = _get_js_template(kind, static=False)

    return execute_js(js,
                      data=data.to_csv(index=False),
                      layout=layout,
                      **kwargs)


def _get_js_template(kind: str, static: bool = True) -> str:
    """Return string template of JS script."""
    script_path = _THIS_DIR / Path(
        f"{['dynamic', 'static'][static]}/templates/{kind}.js")

    return script_path.read_text()
