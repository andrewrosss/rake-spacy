import rake_spacy


def test_version():
    assert rake_spacy.__version__ == "0.2.0"


def test_Rake_is_top_level_attribute():
    assert rake_spacy.Rake


def test_all_modules_are_top_level_attributes():
    assert rake_spacy.aggregators
    assert rake_spacy.cog
    assert rake_spacy.mappers
    assert rake_spacy.phrasers
    assert rake_spacy.rake
    assert rake_spacy.scorers
    assert rake_spacy.stop_tokens
