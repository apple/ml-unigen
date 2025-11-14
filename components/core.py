#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import sys
import time
import random
import logging
import inspect
from pathlib import Path


def get_logger(level=logging.INFO):
    caller_module_name = Path(inspect.stack()[1].filename).stem
    c_format = logging.Formatter('%(asctime)s [%(name)s] %(message)s', "%Y-%m-%d %H:%M:%S")
    _logger = logging.getLogger(caller_module_name)

    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.addFilter(lambda rec: rec.levelno <= logging.INFO)
    stdout_handler.setFormatter(c_format)
    _logger.addHandler(stdout_handler)

    stderr_handler = logging.StreamHandler(stream=sys.stderr)
    stderr_handler.addFilter(lambda rec: rec.levelno > logging.INFO)
    stderr_handler.setFormatter(c_format)
    _logger.addHandler(stderr_handler)

    _logger.setLevel(level)
    return _logger


logger = get_logger()


def init_random_seed(seed: int = None):
    """ Initialize the random seed
    """
    random.seed(seed if seed is not None else time.time())


def backoff(initial_wait: int, attempts: int = 0):
    """Random backoff sleep
    Args:
        initial_wait: in seconds, the initial sleep time
        attempts: exponential backoff retries
    """
    sleep_time = random.uniform(0, initial_wait)
    sleep_time = sleep_time * 2 ** attempts
    logger.info(f"....... sssss ....... ({sleep_time:.1f} seconds)")
    time.sleep(sleep_time)


def backoff_on(max_retries, initial_wait):
    def invisible(func):
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    result_func = func(*args, **kwargs)
                    return result_func
                except Exception as e:
                    delay = (initial_wait * 2 ** retries + random.uniform(0, 1))
                    logger.warning(f"Oops, attempt {retries + 1} failed with: {e}\n"
                                   f"Retrying in {delay:.2f} seconds...")
                    retries += 1
                    time.sleep(delay)
            raise Exception("Max retries reached, aborting...")
        return wrapper
    return invisible
