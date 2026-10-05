"""The demo's key check, which once looked for one name out of three.

brain.nebius accepts a Token Factory key under any of API_KEY_VARS. The demo
used to read only INFERENCE_API_KEY, the second of them, so a key set as
NEBIUS_API_KEY (the name .env.example uses) ran the offline half and said a
key was needed. These tests clear the environment and
stop .env from being read, so they say the same thing on any machine and never
see a real key.
"""

from __future__ import annotations

import dotenv
import pytest

import scripts.demo as demo
from brain import nebius


@pytest.fixture()
def no_keys(monkeypatch):
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
    for var in nebius.API_KEY_VARS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


@pytest.mark.parametrize("var", nebius.API_KEY_VARS)
def test_a_key_under_any_accepted_name_counts(no_keys, var):
    no_keys.setenv(var, "dummy-not-a-key")
    assert demo.have_key() is True


def test_no_key_under_any_name_is_no_key(no_keys):
    assert demo.have_key() is False


def test_a_blank_key_is_not_a_key(no_keys):
    no_keys.setenv(nebius.API_KEY_VARS[0], "   ")
    assert demo.have_key() is False
