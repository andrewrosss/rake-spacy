import pytest
import spacy

from rake_spacy import rake


class TestRake:
    def test_defaults(self):
        r = rake.Rake()

        # constructor args
        assert r.max_length == 100_000
        assert r.min_length == 1
        assert r.nlp.meta["lang"] == "en" and r.nlp.meta["name"] == "core_web_sm"

        # internal attributes
        assert isinstance(r.stop_token_class, rake.stop_tokens.BaseStopTokenIndicator)
        assert isinstance(r.phraser_class, rake.phrasers.BasePhraser)
        assert isinstance(r.token_mapper_class, rake.mappers.BaseTokenMapper)
        assert isinstance(r.word_scorer_class, rake.scorers.BaseScorer)
        assert isinstance(r.aggregator_class, rake.aggregators.BaseAggregator)

    def test_args_passed_to_init_are_set_as_attributes(self, mocker):
        nlp_mock = mocker.MagicMock()
        stop_token_class_mock = mocker.MagicMock()
        phraser_class_mock = mocker.MagicMock()
        token_mapper_class_mock = mocker.MagicMock()
        word_scorer_class_mock = mocker.MagicMock()
        aggregator_class_mock = mocker.MagicMock()

        r = rake.Rake(
            min_length=2,
            max_length=3,
            nlp=nlp_mock,
            stop_token_class=stop_token_class_mock,
            phraser_class=phraser_class_mock,
            token_mapper_class=token_mapper_class_mock,
            word_scorer_class=word_scorer_class_mock,
            aggregator_class=aggregator_class_mock,
        )

        assert r.min_length == 2
        assert r.max_length == 3
        assert r.nlp == nlp_mock
        assert r.stop_token_class == stop_token_class_mock
        assert r.phraser_class == phraser_class_mock
        assert r.token_mapper_class == token_mapper_class_mock
        assert r.word_scorer_class == word_scorer_class_mock
        assert r.aggregator_class == aggregator_class_mock

    @pytest.mark.parametrize(
        "text,expected",
        [
            (
                "red apples, are good in flavour",
                ["red apples", "good", "flavour"],
            ),
            (
                "Apple is looking at buying company for $1 billion",
                ["Apple", "looking", "buying company", "$1 billion"],
            ),
            (
                "Hello, world. Here are two sentences.",
                ["Hello", "world", "two sentences"],
            ),
        ],
    )
    def test__generate_phrases(self, nlp, text, expected):
        r = rake.Rake()
        doc = nlp(text)
        actual = [phrase.text for phrase in r._generate_phrases(doc)]
        assert actual == expected

    @pytest.mark.parametrize(
        "text,expected",
        [
            (
                "red apples, are good in flavour",
                ["good", "flavour"],
            ),
            (
                "Apple is looking at buying company for $1 billion",
                ["Apple", "looking"],
            ),
            (
                "Hello, world. Here are two sentences.",
                ["Hello", "world"],
            ),
        ],
    )
    def test__generate_phrases_with_max_phrase_length_set(self, nlp, text, expected):
        r = rake.Rake(min_length=1, max_length=1)
        doc = nlp(text)
        actual = [phrase.text for phrase in r._generate_phrases(doc)]
        assert actual == expected

    @pytest.mark.parametrize(
        "text,phrase_endpoints,expected",
        [
            (
                "red apples, are better than green apples.",
                [(0, 2), (4, 5), (6, 8)],
                {"apples": 2, "red": 1, "better": 1, "green": 1},
            ),
            (
                "Apple is looking at buying company for $1 billion",
                [(0, 1), (2, 3), (4, 6), (7, 10)],
                {
                    "Apple": 1,
                    "looking": 1,
                    "buying": 1,
                    "company": 1,
                    "$": 1,
                    "1": 1,
                    "billion": 1,
                },
            ),
            (
                "Hello, world. Here are two sentences.",
                [(0, 1), (2, 3), (6, 8)],
                {"Hello": 1, "world": 1, "two": 1, "sentences": 1},
            ),
        ],
    )
    def test__generate_frequency_dist(self, nlp, text, phrase_endpoints, expected):
        r = rake.Rake(token_mapper_class=str)  # type: ignore
        doc = nlp(text)
        phrases = [spacy.tokens.Span(doc, s, e) for s, e in phrase_endpoints]
        actual = r._generate_frequency_dist(phrases)
        assert dict(actual) == expected

    @pytest.mark.parametrize(
        "text,phrase_endpoints,expected",
        [
            (
                "red apples, are better than green apples.",
                [(0, 2), (4, 5), (6, 8)],
                {
                    "apples": {"red": 1, "apples": 2, "green": 1},
                    "red": {"red": 1, "apples": 1},
                    "better": {"better": 1},
                    "green": {"green": 1, "apples": 1},
                },
            ),
            (
                "Apple is looking at buying company for $1 billion",
                [(0, 1), (2, 3), (4, 6), (7, 10)],
                {
                    "Apple": {"Apple": 1},
                    "looking": {"looking": 1},
                    "buying": {"buying": 1, "company": 1},
                    "company": {"buying": 1, "company": 1},
                    "$": {"$": 1, "1": 1, "billion": 1},
                    "1": {"$": 1, "1": 1, "billion": 1},
                    "billion": {"$": 1, "1": 1, "billion": 1},
                },
            ),
            (
                "Hello, world. Here are two sentences.",
                [(0, 1), (2, 3), (6, 8)],
                {
                    "Hello": {"Hello": 1},
                    "world": {"world": 1},
                    "two": {"two": 1, "sentences": 1},
                    "sentences": {"two": 1, "sentences": 1},
                },
            ),
        ],
    )
    def test__generate_word_co_occurance_graph(
        self, nlp, text, phrase_endpoints, expected
    ):
        r = rake.Rake(token_mapper_class=str)  # type: ignore
        doc = nlp(text)
        phrases = [spacy.tokens.Span(doc, s, e) for s, e in phrase_endpoints]
        actual = r._generate_word_co_occurance_graph(phrases)
        assert actual == expected

    @pytest.mark.parametrize(
        "text,expected",
        [
            (
                "red apples, are better than green apples.",
                {"apples": 4, "red": 2, "better": 1, "green": 2},
            ),
            (
                "Apple is looking at buying company for $1 billion",
                {
                    "Apple": 1,
                    "looking": 1,
                    "buying": 2,
                    "company": 2,
                    "$": 3,
                    "1": 3,
                    "billion": 3,
                },
            ),
            (
                "Hello, world. Here are two sentences.",
                {"Hello": 1, "world": 1, "two": 2, "sentences": 2},
            ),
        ],
    )
    def test__generate_degree(self, text, expected):
        r = rake.Rake(token_mapper_class=str)
        # apply() is required because _generate_degree depends on the
        # co_occurange_graph existing as an attribute on r
        _ = r.apply(text)
        actual = r._generate_degree()
        assert actual == expected
