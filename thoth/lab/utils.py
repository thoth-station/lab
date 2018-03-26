"""Various utilities for notebooks."""

from pkgutil import walk_packages
from urllib.parse import urlparse
import importlib
import urllib3

import requests
import pandas as pd


def obtain_location(name: str, verify: bool=True, only_netloc: bool=False) -> str:
    """Obtain location of a service based on it's name in Red Hat's internal network.

    This function basically checks redirect of URL registered at Red Hat's internal network. By doing so it
    is prevented to expose internal URLs. There is queried https://url.corp.redhat.com for redirects.

    >>> obtain_location('thoth-sbu', verify=False)
    """
    # Let's suppress insecure connection warning.
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # Get actual Thoth user API location based on redirect headers.
    response = requests.get(f"https://url.corp.redhat.com/{name}", verify=verify, allow_redirects=False)
    response.raise_for_status()
    location = response.headers['Location']

    if only_netloc:
        return urlparse(location).netloc

    if location.endswith('/'):
        location = location[:-1]

    return location


def display_page(location: str, verify: bool=True, no_obtain_location: bool=False, width: int=980, height: int=900):
    """Display the given page in notebook as iframe."""
    from IPython.display import IFrame

    if not no_obtain_location:
        location = obtain_location(location, verify=verify)

    return IFrame(location, width=width, height=height)


def packages_info(thoth_packages: bool=True):
    """Display information about versions of packages available in the installation."""
    import thoth

    def on_import_error(package_name):
        if thoth_packages and not package_name.startswith('thoth.'):
            return

        packages.append(package_name)
        versions.append(None)
        importable.append(False)

    packages = []
    versions = []
    importable = []
    for pkg in walk_packages(thoth.__path__ if thoth_packages else None, onerror=on_import_error):
        if not pkg.ispkg:
            continue

        name = f'thoth.{pkg.name}' if thoth_packages else pkg.name
        import_successful = False
        version = None

        try:
            module = importlib.import_module(name)
            import_successful = True
            version = module.__version__
        except:
            pass

        packages.append(name)
        versions.append(version)
        importable.append(import_successful)

    return pd.DataFrame(data={'package': packages, 'version': versions, 'importable': importable})
