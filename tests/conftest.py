import pytest
import spacy


@pytest.fixture(scope="session")
def nlp() -> spacy.language.Language:
    return spacy.load("en_core_web_sm")
