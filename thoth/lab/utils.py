#!/usr/bin/env python3
# thoth-lab
# Copyright(C) 2018, 2019 Fridolin Pokorny
#
# This program is free software: you can redistribute it and / or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""Various utilities for notebooks."""
import typing

from functools import partial

from pkgutil import walk_packages
from urllib.parse import urlparse

import importlib
import urllib3

import requests
import pandas as pd


DEFAULT = object()


def obtain_location(name: str, verify: bool = False, only_netloc: bool = False) -> str:
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


def display_page(location: str, verify: bool = True,
                 no_obtain_location: bool = False, width: int = 980, height: int = 900):
    """Display the given page in notebook as iframe."""
    from IPython.display import IFrame

    if not no_obtain_location:
        location = obtain_location(location, verify=verify)

    return IFrame(location, width=width, height=height)


def packages_info(thoth_packages: bool = True) -> pd.DataFrame:
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
        except Exception as e:
            pass

        packages.append(name)
        versions.append(version)
        importable.append(import_successful)

    return pd.DataFrame(data={'package': packages, 'version': versions, 'importable': importable})


def _rhas(fhas, fget, obj: typing.Any, attr: str) -> bool:
    """Recursively check nested attributes of an object.

    :param fhas: callable, function to be used as `hasattr`
    :param fget: callable, function to be used as `getattr`
    :param obj: Any, object to check
    :param attr: str, attribute to find declared by dot notation accessor
    :return: bool, whether the object has the given attribute
    """
    if isinstance(obj, list):
        if not obj:  # empty list
            return False

        return any(_rhas(fhas, fget, item, attr) for item in obj)

    try:
        left, right = attr.split('.', 1)

    except ValueError:
        return fhas(obj, attr)

    return _rhas(fhas, fget, fget(obj, left), right)


def _rget(f,
          obj: typing.Any,
          attr: str,
          repl_missing: typing.Any = None,
          raise_if_missing: bool = False) -> typing.Any:
    """Recursively retrieve nested attributes of an object.

    :param f: callable, function to be used as `getattr`
    :param obj: Any, object to check
    :param attr: str, attribute to find declared by dot notation accessor
    :param repl_missing: bool, whether to raise on missing attribute
    :param raise_if_missing:
    :return: Any, retrieved attribute
    """
    if isinstance(obj, (list, set)):
        if len(obj) <= 0:
            return None

        return [
            _rget(f, item, attr, repl_missing, raise_if_missing)
            for item in obj
        ]

    right = ''
    attrs = attr.split('.', 1)

    if not attrs:
        return obj
    elif len(attrs) == 2:
        left, right = attr.split('.', 1)
    else:
        left = attr

    if not right:
        try:
            return f(obj, attr)

        except (AttributeError, KeyError) as exc:
            if raise_if_missing:
                raise exc

            return repl_missing

    return _rget(f, f(obj, left), right, repl_missing, raise_if_missing)


def has(obj, attr):
    """Combine both `hasattr` and `in` into universal `has`."""
    def _in(_obj, _attr):
        try:
            return _attr in _obj
        except TypeError:
            # object is not iterable
            return False

    return any([hasattr(obj, attr), _in(obj, attr)])


def get(obj, attr, *, default: typing.Any = DEFAULT):
    """Combine both `getattr` and `dict.get` into universal `get`."""
    _getattr = getattr if default is DEFAULT else partial(getattr, default=default)
    _get = dict.get if default is DEFAULT else partial(dict.get, default=default)

    try:
        return _getattr(obj, attr)

    except AttributeError as exc:
        if isinstance(obj, dict):
            return _get(obj, attr, default, )
        elif default is not DEFAULT:
            return default

        raise exc


# syntactic sugar to _rhas and _rget which is meant for users

rhasattr = partial(_rhas, hasattr, getattr)
rhasattr.__doc__ = _rhas.__doc__
rgetattr = partial(_rget, getattr)
rgetattr.__doc__ = _rget.__doc__

rhas = partial(_rhas, has, get)
rhasattr.__doc__ = _rhas.__doc__
rget = partial(_rget, get)
rgetattr.__doc__ = _rget.__doc__
