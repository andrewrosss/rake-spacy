# stdlib
# ------

# 3rd party
import pytest
import spacy

# local
# -----


@pytest.fixture(scope="session")
def nlp():
    return spacy.load("en_core_web_sm")
