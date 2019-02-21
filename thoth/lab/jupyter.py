"""Tools for Jupyter notebook."""


from IPython.core.display import display, Javascript


def link_css(stylesheet: str):
    """Link CSS stylesheet."""
    script = (
        f"const href = \"{stylesheet}\";"
        """
        var link = document.createElement("link");
        link.rel = "stylesheet";
        link.type = "text/css";
        link.href = href;
        
        document.head.appendChild(link);
        """
    )

    return display(Javascript(script))


def link_js(lib: str):
    """Link JavaScript library."""
    script = (
        f"const src = \"{lib}\";"
        """
        var script = document.createElement("script");
        script.src = src;
        
        document.head.appendChild(script);
        """
    )

    return display(Javascript(script))


def load_style(style: str):
    """Create new style element and add it to the page."""

    script = (
        f"const style = `{style}`;"
        """
        var e = document.createElement(\"style\");
        $(e).html(`${style}`).attr('type', 'text/css');
        
        document.head.appendChild(e);
        """
    )

    return display(Javascript(script))


def load_script(script: str):
    """Create new script element and add it to the page."""

    script = (
        f"const script = `{script}`;"
        """
        var e = document.createElement(\"script\");
        $(e).html(`${style}`).attr('type', 'text/javascript');
        
        document.head.appendChild(e);
        """
    )

    return display(Javascript(script))
