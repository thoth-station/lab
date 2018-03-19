"""Various utilities for notebooks."""

import requests
import urllib3
from urllib.parse import urlparse


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
