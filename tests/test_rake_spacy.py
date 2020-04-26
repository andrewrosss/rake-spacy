# stdlib
# ------

# 3rd party
# ---------

# local
import rake_spacy


def test_version():
    assert rake_spacy.__version__ == "0.1.0"


def test_Rake_is_top_level_attribute():
    assert rake_spacy.Rake


def test_Metric_is_top_level_attribute():
    assert rake_spacy.Metric
