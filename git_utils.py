import subprocess
from urllib.parse import urlparse

def get_git_revision_short_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()


def get_git_url() -> str:
    basic_url = subprocess.check_output(['git', 'config', '--get', 'remote.origin.url']).decode('ascii').strip()
    parsed = urlparse(basic_url)
    if parsed.username and parsed.password:
        # erase username and password
        return parsed._replace(netloc="{}".format(parsed.hostname)).geturl()
    else:
        return parsed.geturl()

import hashlib
import os
import validators
import requests
import logging

CACHE_FOLDER = os.path.join(os.environ.get('CACHE_FOLDER', '.cache'))

def is_cached_file(resource: str, cache_folder=CACHE_FOLDER, enforce_ending='')-> bool:
    md5 = hashlib.md5(resource.encode('utf-8')).hexdigest()
    return os.path.isfile(os.path.join(cache_folder, md5 + enforce_ending))

def is_cached_folder(resource: str, cache_folder=CACHE_FOLDER)-> bool:
    md5 = hashlib.md5(resource.encode('utf-8')).hexdigest()
    return os.path.isdir(os.path.join(cache_folder, md5))

def cached_file(resource: str, cache_folder=CACHE_FOLDER, enforce_ending='')-> str:
    md5 = hashlib.md5(resource.encode('utf-8')).hexdigest()
    if is_cached_file(resource, cache_folder=cache_folder, enforce_ending=enforce_ending):
        # return the cached file resource
        return os.path.join(cache_folder, md5 + enforce_ending)
    elif os.path.isfile(resource):
        # return the persistent file
        return resource
    elif validators.url(resource):
        file_path = os.path.join(cache_folder, md5 + enforce_ending)
        logging.info(f"Download file: {resource} to {file_path}")
        try:
            # download from url into cached
            r = requests.get(resource, allow_redirects=True)
            # write to cache
            with open(file_path, 'wb') as output:
                output.write(r.content)
        except:
            # something went wrong --> we have to remove the file again to keep a valid cache
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        # return file
        return file_path
