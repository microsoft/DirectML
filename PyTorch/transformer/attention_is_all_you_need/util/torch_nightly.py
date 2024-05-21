"""
Return a list of recent PyTorch wheels published on download.pytorch.org.
Users can specify package name, python version, platform, and the number of days to return.
If one of the packages specified is missing on one day, the script will skip outputing the results on that day.
"""

import os
import re
import requests
import argparse
import urllib.parse
from datetime import date, timedelta
from bs4 import BeautifulSoup
from collections import defaultdict

torch_wheel_nightly_base ="https://download.pytorch.org/whl/nightly/cu113/"
torch_nightly_wheel_index = "https://download.pytorch.org/whl/nightly/cu113/torch_nightly.html"
torch_nightly_wheel_index_override = "torch_nightly.html" 

def memoize(function):
    """ 
    """
    call_cache = {}

    def memoized_function(*f_args):
        if f_args in call_cache:
            return call_cache[f_args]
        call_cache[f_args] = result = function(*f_args)
        return result

    return memoized_function

@memoize
def get_wheel_index_data(py_version, platform_version, url=torch_nightly_wheel_index, override_file=torch_nightly_wheel_index_override):
    """
    """
    if os.path.isfile(override_file) and os.stat(override_file).st_size:
        with open(override_file) as f:
            data = f.read()
    else:
        r = requests.get(url)
        r.raise_for_status()
        data = r.text
    soup = BeautifulSoup(data, 'html.parser')
    data = defaultdict(dict)
    for link in soup.find_all('a'):
        group_match = re.search("([a-z]*)-(.*)-(.*)-(.*)-(.*)\.whl", link.text)
        # some packages (e.g., torch-rec) doesn't follow this naming convention
        if not group_match:
            continue
        pkg, version, py, py_m, platform = group_match.groups()
        version = urllib.parse.unquote(version)
        if py == py_version and platform == platform_version:
            full_url = os.path.join(torch_wheel_nightly_base, link.text)
            data[pkg][version] = full_url
    return data

def get_nightly_wheel_urls(packages:list, date:date,
                           py_version='cp38', platform_version='linux_x86_64'):
    """Gets urls to wheels for specified packages matching the date, py_version, platform_version
    """
    date_str = f"{date.year}{date.month:02}{date.day:02}"
    data = get_wheel_index_data(py_version, platform_version)

    rc = {}
    for pkg in packages:
        pkg_versions = data[pkg]
        # multiple versions could happen when bumping the pytorch version number
        # e.g., both torch-1.11.0.dev20220211%2Bcu113-cp38-cp38-linux_x86_64.whl and
        # torch-1.12.0.dev20220212%2Bcu113-cp38-cp38-linux_x86_64.whl exist in the download link
        keys = sorted([key for key in pkg_versions if date_str in key], reverse=True)
        if len(keys) > 1:
            print(f"Warning: multiple versions matching a single date: {keys}, using {keys[0]}")
        if len(keys) == 0:
            return None
        full_url = pkg_versions[keys[0]]
        rc[pkg] = {
                        "version": keys[0],
                        "wheel": full_url,
        }
    return rc

def get_nightly_wheels_in_range(packages:list, start_date:date, end_date:date,
                                py_version='cp38', platform_version='linux_x86_64', reverse=False):
    rc = []
    curr_date = start_date
    while curr_date <= end_date:
        curr_wheels = get_nightly_wheel_urls(packages, curr_date,
                                             py_version=py_version,
                                             platform_version=platform_version)
        if curr_wheels is not None:
            rc.append(curr_wheels)
        curr_date += timedelta(days=1)
    if reverse:
        rc.reverse()
    return rc

def get_n_prior_nightly_wheels(packages:list, n:int,
                               py_version='cp38', platform_version='linux_x86_64', reverse=False):
    end_date = date.today()
    start_date = end_date - timedelta(days=n)
    return get_nightly_wheels_in_range(packages, start_date, end_date,
                                       py_version=py_version, platform_version=platform_version, reverse=reverse)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyver", type=str, default="cp38", help="PyTorch Python version")
    parser.add_argument("--platform", type=str, default="linux_x86_64", help="PyTorch platform")
    parser.add_argument("--priordays", type=int, default=1, help="Number of days")
    parser.add_argument("--reverse", action="store_true", help="Return reversed result")
    parser.add_argument("--packages", required=True, type=str, nargs="+", help="List of package names")
    args = parser.parse_args()
    wheels = get_n_prior_nightly_wheels(packages=args.packages,
                                        n=args.priordays,
                                        py_version=args.pyver,
                                        platform_version=args.platform,
                                        reverse=args.reverse)
    for wheelset in wheels:
        for pkg in wheelset:
            print(f"{pkg}-{wheelset[pkg]['version']}: {wheelset[pkg]['wheel']}")
