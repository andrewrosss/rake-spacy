# stdlib
# ------

# 3rd party
import pytest
import spacy

# local
from rake_spacy import rake


class TestMetric:
    def test_enumeration_members(self):
        assert isinstance(rake.Metric.DEGREE_TO_FREQUENCY_RATIO, rake.Metric)
        assert isinstance(rake.Metric.WORD_DEGREE, rake.Metric)
        assert isinstance(rake.Metric.WORD_FREQUENCY, rake.Metric)


class TestRake:
    def test_defaults(self):
        r = rake.Rake()

        # constructor args
        assert r.ranking_metric == rake.Metric.DEGREE_TO_FREQUENCY_RATIO
        assert r.max_length == 100_000
        assert r.min_length == 1
        assert r.nlp is None

        # internal attributes
        assert r.frequency_dist is None
        assert r.degree is None
        assert r.co_occurance_graph is None
        assert r.rank_list is None
        assert r.ranked_phrases is None

    def test_user_specified_args_to_init(self, nlp):
        r = rake.Rake(
            nlp=nlp,
            ranking_metric=rake.Metric.WORD_DEGREE,
            max_length=10,
            min_length=10,
        )

        assert r.ranking_metric == rake.Metric.WORD_DEGREE
        assert r.max_length == 10
        assert r.min_length == 10
        assert r.nlp == nlp

    @pytest.mark.parametrize(
        "text,expected",
        [
            (
                "red apples, are good in flavour",
                [False, False, True, True, False, True, False],
            ),
            (
                "Apple is looking at buying company for $1 billion",
                [False, True, False, True, False, False, True, False, False, False],
            ),
            (
                "Hello, world. Here are two sentences.",
                [False, True, False, True, True, True, False, False, True],
            ),
        ],
    )
    def test_to_ignore(self, nlp, text, expected):
        r = rake.Rake()
        doc = nlp(text)
        actual = [r.to_ignore(t) for t in doc]
        assert actual == expected, [
            (t, e, a) for t, e, a in zip(doc, expected, actual) if e != a
        ]

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("red apples, are good in flavour", ["red apples", "good", "flavour"],),
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
    def test_generate_phrases(self, nlp, text, expected):
        r = rake.Rake()
        doc = nlp(text)
        actual = [phrase.text for phrase in r.generate_phrases(doc)]
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
    def test_generate_frequency_dist(self, nlp, text, phrase_endpoints, expected):
        r = rake.Rake()
        doc = nlp(text)
        phrases = [spacy.tokens.Span(doc, s, e) for s, e in phrase_endpoints]
        actual = r.generate_frequency_dist(phrases)
        assert actual == expected
        assert r.frequency_dist == expected

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
    def test_generate_word_co_occurance_graph(
        self, nlp, text, phrase_endpoints, expected
    ):
        r = rake.Rake()
        doc = nlp(text)
        phrases = [spacy.tokens.Span(doc, s, e) for s, e in phrase_endpoints]
        actual = r.generate_word_co_occurance_graph(phrases)
        assert actual == expected
        assert r.co_occurance_graph == expected

    @pytest.mark.parametrize(
        "text,phrase_endpoints,expected",
        [
            (
                "red apples, are better than green apples.",
                [(0, 2), (4, 5), (6, 8)],
                {"apples": 4, "red": 2, "better": 1, "green": 2},
            ),
            (
                "Apple is looking at buying company for $1 billion",
                [(0, 1), (2, 3), (4, 6), (7, 10)],
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
                [(0, 1), (2, 3), (6, 8)],
                {"Hello": 1, "world": 1, "two": 2, "sentences": 2},
            ),
        ],
    )
    def test_generate_degree(self, nlp, text, phrase_endpoints, expected):
        r = rake.Rake()
        doc = nlp(text)
        phrases = [spacy.tokens.Span(doc, s, e) for s, e in phrase_endpoints]
        actual = r.generate_degree(phrases)
        assert actual == expected
        assert r.degree == expected

    @pytest.mark.parametrize("degree", range(3))
    @pytest.mark.parametrize("frequency_dist", range(1, 3))
    def test_word_score_degree_to_frequency_ratio(self, degree, frequency_dist):
        r = rake.Rake(ranking_metric=rake.Metric.DEGREE_TO_FREQUENCY_RATIO)
        r.degree = {"test": degree}
        r.frequency_dist = {"test": frequency_dist}

        actual = r.word_score("test")
        expected = degree / frequency_dist

        assert actual - expected < 1e-7

    @pytest.mark.parametrize("degree", range(3))
    @pytest.mark.parametrize("frequency_dist", range(1, 3))
    def test_word_score_word_degree(self, degree, frequency_dist):
        r = rake.Rake(ranking_metric=rake.Metric.WORD_DEGREE)
        r.degree = {"test": degree}
        r.frequency_dist = {"test": frequency_dist}

        actual = r.word_score("test")
        expected = degree

        assert actual - expected < 1e-7

    @pytest.mark.parametrize("degree", range(3))
    @pytest.mark.parametrize("frequency_dist", range(1, 3))
    def test_word_score_word_frequency(self, degree, frequency_dist):
        r = rake.Rake(ranking_metric=rake.Metric.WORD_FREQUENCY)
        r.degree = {"test": degree}
        r.frequency_dist = {"test": frequency_dist}

        actual = r.word_score("test")
        expected = frequency_dist

        assert actual - expected < 1e-7
