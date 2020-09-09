import pytest
import spacy


@pytest.fixture(scope="session")
def nlp():
    return spacy.load("en_core_web_sm")
