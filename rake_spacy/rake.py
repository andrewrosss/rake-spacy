# stdlib
import enum
import itertools
import collections
from typing import Callable, DefaultDict, Dict, List, Tuple, Union

# 3rd party
import spacy

# local
# -----


class Metric(enum.Enum):
    """Different metrics that can be used for ranking."""

    DEGREE_TO_FREQUENCY_RATIO = 0  # uses d(w)/f(w) as the metric
    WORD_DEGREE = 1  # uses d(w) as the metric
    WORD_FREQUENCY = 2  # uses f(w)


class Rake:
    """Rapid Automatic Keyword Extraction Algorithm.

    Args:
        nlp (spacy.lang.en.English, optional): The spacy language model to
            use to parse/tokenize the given text. If ``None`` then one is
            created from the ``en_core_web_sm`` model. Defaults to None.
        min_length (int, optional): The minimum length (inclusive) for a
            phrase to be included in the key-phrase computation. Phrases
            shorter than this are excluded. Defaults to 1.
        max_length (int, optional): The maximum length (inclusive) for a
            phrase to be included in the key-phrase computation. Phrases
            longer than this are excluded. Defaults to 100_000.
        token_key (str, optional): The token attribute to use in the
            key-phrase computation. For example, if ``token_key=lemma_``
            then the words 'Whale', 'whale', and 'whales' will all be mapped
            to the same node in the co-occurence graph, whereas if
            ``token_key=text`` then those three words will be distinct nodes
            in the underlying graph. Defaults to "lemma_".
        ranking_metric (Metric, optional): The metric to use to rank/score
            words. Defaults to Metric.WORD_FREQUENCY.
        use_default_ignorables (bool, optional): Whether to ignore the
            "default" stop words. Defaults to True. The "default" stop words
            are considered to be any token for which the following expression
            evaluates to true:

            .. code-block:: python

               # t is a spacy.tokens.Token
               (t.is_stop or t.is_space or t.is_punct) and not t.like_num

        phrase_fn (Union[str, Callable], optional): This callable should
            accept a ``Doc`` and produce a list of spacy ``Span`` objects.
            This callable will be used to generate candidate key-phrases. This
            argument can also be a string (as per the default value), in which
            case it should be an existing method on this object (i.e. one of:
            "ent_and_noun_chunk_phrases" or "contiguous_non_stop_phrases").
            Defaults to "ent_and_noun_chunk_phrases".
        ignore_fn (Callable, optional): This callable should accept a ``Token``
            and produce a boolean indicating whether that token should be
            ignored. It is called on tokens before registering them in the
            co-occurance graph. Defaults to None.
        score_agg_fn (Callable, optional): This callable should accept a
            list of floats and produce a float. This function is given a list
            of floats representing the word scores (as determined by
            ``ranking_metric``) for the words in a given phrase, and should
            return a float which represents the desired aggregate score for
            that phrase. If ``None`` then the sum of the word scores is used.
            Defaults to None.

    Note:
        To emulate the "original" RAKE algorithm. Instantiate the object
        as follows:

        .. code-block:: python

           r = Rake(
               token_key='text',
               ranking_metric=Metric.DEGREE_TO_FREQUENCY_RATIO,
               phrase_fn='contiguous_non_stop_phrases',
           )
           r.extract_keywords_from_text(...)
    """

    def __init__(
        self,
        nlp=None,
        min_length: int = 1,
        max_length: int = 100_000,
        token_key: str = "lemma_",
        ranking_metric: Metric = Metric.WORD_FREQUENCY,
        use_default_ignorables: bool = True,
        phrase_fn: Union[str, Callable] = "ent_and_noun_chunk_phrases",
        ignore_fn: Callable = None,
        score_agg_fn: Callable = None,
    ) -> None:

        self.nlp = spacy.load("en_core_web_sm") if nlp is None else nlp
        self.max_length = max_length
        self.min_length = min_length
        self.token_key = token_key
        self.ranking_metric = ranking_metric
        self.use_default_ignorables = use_default_ignorables
        self.phrase_fn = phrase_fn if callable(phrase_fn) else getattr(self, phrase_fn)
        self.ignore_fn = ignore_fn
        self.score_agg_fn = sum if score_agg_fn is None else score_agg_fn

        # stuff to be extracted from the provided text
        self.frequency_dist = None
        self.degree = None
        self.co_occurance_graph = None
        self.rank_list = None
        self.ranked_phrases = None

    def extract_keywords_from_text(
        self, text: str
    ) -> List[Tuple[float, spacy.tokens.Span]]:
        """Extracts keywords/phrases from a string of text.

        Args:
            text (str): The text from which keywords/phrases should be
                extracted.

        Returns:
            List[Tuple[float, spacy.tokens.Span]]: A list of (score, phrase)
            pairs.
        """
        doc = self.nlp(text)
        phrases = self.generate_phrases(doc)
        self.generate_frequency_dist(phrases)
        self.generate_degree(phrases)
        return self.generate_ranklist(phrases)

    def generate_phrases(self, doc: spacy.tokens.Doc) -> List[spacy.tokens.Span]:
        """Generates phrases (spans) from a document.

        This method extracts phrases from the function specified by the
        ``phrase_fn`` attribute. The phrases produced by that function are
        then validated againsts the phrase length constraints
        (``self.min_length`` and ``self.max_length``).

        Args:
            doc (spacy.tokens.Doc): The document from which to extract
                phrases.

        Returns:
            List[spacy.tokens.Span]: The extracted phrases.
        """
        phrases = self.phrase_fn(doc)
        return [p for p in phrases if self.min_length <= len(p) <= self.max_length]

    def contiguous_non_stop_phrases(
        self, doc: spacy.tokens.Doc
    ) -> List[spacy.tokens.Span]:
        """Produces a list of spans where each span is a list of contiguous
        non-stopwords.

        Note:
            This is the (essentially) the phrase-generation technique proposed
            in the original RAKE paper (DOI: 10.1002/9780470689646.ch1)

        Args:
            doc (spacy.tokens.Doc): The document from which to generate the
            phrases.

        Returns:
            List[spacy.tokens.Span]: A list of candidate phrases.
        """
        groups = [
            list(g)
            for k, g in itertools.groupby(doc, lambda t: not self.to_ignore(t))
            if k
        ]
        return [doc[g[0].i : g[-1].i + 1] for g in groups]

    def ent_and_noun_chunk_phrases(
        self, doc: spacy.tokens.Doc
    ) -> List[spacy.tokens.Span]:
        """Produces a list of spans where each span is either an named entity, a
        noun chunk, or if an entity and noun-chunk intersect then the "larger"
        of the two.

        Args:
            doc (spacy.tokens.Doc): The document from which to generate the phrases.

        Returns:
            List[spacy.tokens.Span]: A list of candidate phrases.
        """
        return spacy.util.filter_spans(list(doc.ents) + list(doc.noun_chunks))

    def to_ignore(self, t: spacy.tokens.Token) -> bool:
        """Determines of a token should be ignored.

        Args:
            t (spacy.tokens.Token): The token under consideration.

        Returns:
            bool: Whether this token should be ignored of not.
        """
        outcome = False

        if self.use_default_ignorables:
            outcome = outcome or (
                (t.is_stop or t.is_space or t.is_punct) and not t.like_num
            )

        if self.ignore_fn is not None:
            outcome = outcome or self.ignore_fn(t)

        return outcome

    def generate_frequency_dist(
        self, phrases: List[spacy.tokens.Span]
    ) -> Dict[str, int]:
        """Generates a dict-like object (specifically, a
        ``collections.Counter``) whose keys are the words in phrases and
        whose values are the counts of those words among the list of provided
        phrases.

        Tokens are first filtered using the ``.to_ignore()`` method, from
        there tokens are converted to strings using the ``Token`` attribute
        referenced by the ``.token_key`` string (on this object). That is,
        if ``self.token_key = 'lemma_'``, then each token is converted to a
        string using its ``Token.lemma_`` attribute.

        Args:
            phrases (List[spacy.tokens.Span]): The phrases from which the
                words are counted.

        Returns:
            Dict[str, int]: The mapping from words to counts.
        """
        words = [t for t in itertools.chain(*phrases) if not self.to_ignore(t)]
        word_strings = [self.to_string(t) for t in words]
        fd = collections.Counter(word_strings)
        self.frequency_dist = fd
        return fd

    def generate_degree(self, phrases: List[spacy.tokens.Span]) -> Dict[str, int]:
        """Generates a dict-like object (specifically, a ``defaultdict``)
        where each key is a token string and each value is the associated
        degree of that word (node) in the co-occurance graph.

        Args:
            phrases (List[spacy.tokens.Span]): The phrases from which the
                word degrees are computed.

        Returns:
            Dict[str, int]: The mapping of words to degrees

        Note:
            This method calls ``.generate_word_co_occurange_graph()`` under
            the hood.
        """
        cog = self.generate_word_co_occurance_graph(phrases)
        degree = collections.defaultdict(int)
        for word in cog:
            degree[word] = sum(cog[word].values())
        self.degree = degree
        return degree

    def generate_word_co_occurance_graph(
        self, phrases: List[spacy.tokens.Span]
    ) -> Dict[str, DefaultDict[str, int]]:
        """Generates the co-occurance graph for the words in the given list
        of phrases.

        Args:
            phrases (List[spacy.tokens.Span]): The phrases to use in computing
                the graph.

        Returns:
            Dict[str, DefaultDict[str, int]]: The co-occurance graph.
        """
        cog = collections.defaultdict(lambda: collections.defaultdict(int))
        for phrase in phrases:
            for word, coword in itertools.product(phrase, phrase):
                if not self.to_ignore(word) and not self.to_ignore(coword):
                    word_string = self.to_string(word)
                    coword_string = self.to_string(coword)
                    cog[word_string][coword_string] += 1
        self.co_occurance_graph = cog
        return cog

    def generate_ranklist(
        self, phrases: List[spacy.tokens.Span]
    ) -> List[Tuple[float, spacy.tokens.Span]]:
        """Returns a sorted list of (phrase-score, phrase) pairs.

        Args:
            phrases (List[spacy.tokens.Span]): The phrases to rank/score.

        Returns:
            List[Tuple[float, spacy.tokens.Span]]: The list of (phrase-score,
            phrase) pairs.
        """
        scores = [([self.word_score(t) for t in p], p) for p in phrases]
        rank_list = [(self.score_agg_fn(word_scores), p) for word_scores, p in scores]
        self.rank_list = sorted(rank_list, key=lambda t: (-t[0], t[1].text))
        self.ranked_phrases = [p for _, p in self.rank_list]
        return self.rank_list

    def word_score(self, t: spacy.tokens.Token) -> float:
        """Scores a token.

        Args:
            t (spacy.tokens.Token): The token to score.

        Returns:
            float: ``t``'s score as determined by ``ranking_metric``.
        """
        word = self.to_string(t)
        if self.ranking_metric == Metric.DEGREE_TO_FREQUENCY_RATIO:
            if self.frequency_dist[word] > 0:
                score = self.degree[word] / self.frequency_dist[word]
            else:
                score = 0
        elif self.ranking_metric == Metric.WORD_DEGREE:
            score = self.degree[word]
        else:
            score = self.frequency_dist[word]
        return 1.0 * score

    def to_string(self, t: spacy.tokens.Token) -> str:
        """Converts a token to a string.

        Args:
            t (spacy.tokens.Token): The token to convert.

        Returns:
            str: A string representation of the token.

        Note:
            This need **NOT** be the same as calling ``str(t)``.
        """
        return getattr(t, self.token_key)
