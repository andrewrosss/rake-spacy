# rake-spacy

![Code Style](https://github.com/andrewrosss/rake-spacy/workflows/Code%20Style/badge.svg) ![Tests](https://github.com/andrewrosss/rake-spacy/workflows/Tests/badge.svg) [![codecov](https://codecov.io/gh/andrewrosss/rake-spacy/branch/master/graph/badge.svg)](https://codecov.io/gh/andrewrosss/rake-spacy)

Python implementation of the RAKE (short for **R**apid **A**utomatic **K**eyword **E**xtraction) algorithm using spaCy.

## Installation

```bash
pip install rake-spacy
```

Since rake-spacy depends on spacy, and to used spacy one has to load a language model, by default, rake-spacy will try to load spacy's `en_core_web_sm` model, so also grab that language model as well.

```bash
python -m spacy download en_core_web_sm
```
While this is the model used by rake-spacy by default, you can easily provide rake spacy with any language model/pipeline of your chosing. (Just about any `nlp` object from the spacy docs.)

## Getting Started

To quickly extract some ranked keywords

```python
from rake_spacy import Rake

r = Rake()

text = "Compatibility of systems of linear constraints over the set of natural numbers. Criteria of compatibility of a system of linear Diophantine equations, strict inequations, and nonstrict inequations are considered. Upper bounds for components of a minimal set of solutions and algorithms of construction of minimal generating sets of solutions for all types of systems are given. These criteria and the corresponding algorithms for constructing a minimal supporting set of solutions can be used in solving all the considered types of systems and systems of mixed types."

ranklist = r.apply(text)

print(ranklist)  # [(8.0, minimal generating sets), (8.0, minimal supporting set), (7.0, minimal set), (5.0, considered types), (5.0, system), (5.0, systems), ...]
```

## Differnent Language Models

To specify a language model other than spacy's `en_core_web_sm` model, you simply instatiate the language model of your choosing, and pass it to `Rake` in instantiation:

```python
from rake_spacy import Rake
import spacy

nlp = spacy.load('en_core_web_lg')  # assuming this model is installed
r = Rake(nlp=nlp)

text = 'This is a sentence.'

# Rake is now using the provided nlp object to prase the document
ranklist = r.apply(text)
```

## Code Structure and Customization

At it's core, the RAKE algorithm is conceptually simple:

1. **Extract candidate spans** (sequences of contiguous words/tokens) from a piece of text.
1. **Create the co-occurance graph** based on the spans extracted from the previous step (represented as an adjacency matrix, where two words/tokens are "adjacent", i.e. share an edge in the graph, if they are found in the same span).
1. **Compute a score for each word/token** based from the co-occurance graph.
1. **Compute a score for each span** by aggregating word scores (determined in the previous step)
1. **Order the spans** based on the aggregate span-scores.

There is a fairly close mapping between the highlevel steps outlined above, the package structure of rake-spacy, and the available keyword arguments of the `Rake` object. To understand the structure of the code we'll start with the first step.

## Phrasers

To apply RAKE to a piece of text the first thing we need is a way to extract candidate spans from the text. Throughout the remainder of the documentation these spans will be referred to as "phrases".

By default, rake-spacy extracts "contiguous non-stop-words" as the candidate phrases. What are "contiguous non-stop-words"? These are just chunks of text (words/tokens that are side-by-side) which contain no stop words. (More on how stop words/tokens are determined below.)

This process of extracting candidate phrases for RAKE to consider (i.e. from which to build the co-occurance graph) can be customized by providing a user-specified callable as the `phraser_class` parameter when instantiating `Rake`:

```python
def my_phraser_func(doc: spacy.language.Language) -> List[spacy.tokens.Span]:
    ...  # slice and dice the document as desired
    return my_list_of_spans

r = Rake(phraser_class=my_phraser_func)
```

Essentially, the object passed as the value to the `phraser_class` argument must be a callable which accepts a `spacy.tokens.Doc` object and returns a list of `spacy.token.Span` objects. Those span objects will be the phrases used to construct the co-occurance graph.

Phraser classes included in this package can be found in the `rake_spacy.phrasers` module.

### Aside #1

Why call the paramter phraser\_<b>class</b>? This is because the "batteries-included" phraser callables (found in the `rake_spacy.phrasers` module) inherit from `rake_spacy.phrasers.BasePhraser`. The choice to use classes is to allow for "parameterizable callables" which still only take one argument when called. Specifically, when `Rake` calls the `phraser_class` callable the only parameter that is provided is the spacy `doc`. By using a class to define this callable additional parameters can be specified in the class's `__init__` method, and used in the `__call__` method.

Of course, there are other ways to achieve this same "parameterizability", one could define the phraser parameters in the global scope and then reference those variables in the function body.

```python
GLOBAL_NAME = 'World'

def f(greeting):
    print(f'{greeting}, {GLOBAL_NAME}!')
```

Or if global variables aren't your thing you could use `functools.partial`

```python
import functools

def full_g(greeting, name):
    print(f'{greeting}, {name}')

g = functools.partial(full_g, name='World')
```

As with most things in python there's typically more than one way to do something. "Callable classes" seemed like a straight forward choice, and are found throughout the code.

## (Word) Scorers

Skipping over the construction of the co-occurance graph for a moment. The next thing to do is to score each of the words/tokens found in the collection of phrases using the information contained in the co-occurance graph. By default, the score rake-spacy attributes to each word/token is simply the number of times it occured in the text (these are the diagonal elements in the co-occurance graph).

To override the default behaviour we simply define a callable that takes the co-occurance graph and a token (vertex in the graph) and returns a numeric score, and then provide that callable as the value to the `word_scorer_class` argument:

```python
def my_scorer_func(
    cograph: DefaultDict[str, DefaultDict[str, int]],
    token: spacy.tokens.Token
) -> float:
    ...  # compute the score for this token using the co-ocrrance graph
    return token_score

r = Rake(word_scorer_class=my_scorer_func)
```

Word-scorer classes included in this package can be found in the `rake_spacy.scorers` module.

### Aside #2

It's worth mentioning here that the co-occurance graph is stored as (essentially) a defaultdict within a defaultdict. Specifically, "indexing" once into the co-occurance graph with a token, `cograph[token]`, returns a dict-like object which maps co-occuring words/tokens to co-occurance counts. That is, `cograph[token][cotoken]` is an integer denoting the number of time `token` and `cotoken` appeared in the same phrase.

`cograph[token]` looks weird, you might be thinking. `token` is a `spacy.tokens.Token` object and it's being used as the key to this dict-looking-thing? Of course, `spacy.tokens.Token` objects have a `__hash__` method, and so they _can_ be used as dictionary keys. But that's not (necessarily) what is going on under the hood (although it _could_ be - if you really wanted it to be that way). Using tokens as "keys" is facilitated (and, in fact, is customizable as well) by another set of classes we have yet to talk about, but this small detail is precisely why you can think of the co-occurance graph as being **_essentially_** like a defaultdict within a defaultdict. In a nut shell, yes, you should index into the co-occurance graph with `spacy.tokens.Token` objects, directly. More on this below.

## Aggregators

OK, so we've split our text up into phrases, computed the co-occurance graph and scored the words/tokens, now we have to score the phrases themselves.

By default, rake-spacy will just sum up all the token scores in each phrase. But this default behaviour can be changed by providing a callable as the value to the `aggregator_class` keyword argument.

```python
def my_aggregator_func(word_scores: List[float]) -> float:
    ...  # compute the phrase-score from the word-scores of its constiuent words
    return phrase_score

r = Rake(aggregator_class=my_aggregator_func)
```

Aggregator classes included in this package can be found in the `rake_spacy.aggregators` module.

This covers probably the most important parts of how one might want to customize how the RAKE algorithm is administered to a piece of text, but we did skim over a few things. We'll cover those next.

## Stop-Words/Tokens

Spacy is a fantastic library and with basically no code at all we can breakdown a piece of text into fine-grained parts seemingly "for free". However, when applying the RAKE algorithm we'd probably like to ignore some of the more common or less-meaningful words/tokens. Stop-words were alluded to above, how can we customize which words/tokens are considered "stop-words" and should effectively be ignored?

By default, rake-spacy assumes that any token for which the following expression evaluates to `True` is a stop-token:

```python
(token.is_stop or token.is_space or token.is_punct) and not token.like_num
```

To customize this behaviour we can implement (you guessed it) a _callable_ which accepts a token and returns a boolean. (True if this token should be ignored, False if it should be kept.)

```python
def my_stop_word_detector_func(token: spacy.tokens.Token) -> bool:
    ...  # determine if this toke is a stop-token (True) or not (False)
    return is_stop_token

r = Rake(stop_token_class=my_stop_word_detector_func)
```

Internally, rake-spacy will use this callable on each token (actually, pair of tokens) before updating the co-occurance graph. If the callable indicates that one of the two tokens is a stop-word/token and should be ignored, that pair is skipped and rake-spacy moves onto the next pair.

One thing to note is that a user-specified stop-word callable will likely not have a role in how phrases are extracted (because "phrasing" is also user-specifiable). As such, if you provide `Rake` with a `stop_token_class` override, for consistency, you may want to use that same stop-word callable in the phraser callable. Perhaps something like:

```python
class MyStopWordIndicator(rake_spacy.stop_words.BaseStopTokenIndicator):
    def __call__(self, token):
        ...  # your implementation

class MyPhraser(rake_spacy.phrasers.BasePhraser):
    def __init__(self, stop_word_indicator):
        self.stop_word_indicator = stop_word_indicator

    def __call__(self, doc):
        ...  # your implementation, using self.stop_word_indicator

stop_token_class = MyStopWordIndicator()
phraser_class = MyPhraser(stop_token_class)

r = Rake(stop_token_class=stop_token_class, phraser_class=phraser_class)
```

This is actually what rake-spacy does in the `Rake.__init__()` method.

Stop-word/token classes included in this package can be found in the `rake_spacy.stop_words` module.

## The Co-Occurance Graph

So, why is the co-occurance graph **_essentially_** like a defaultdict within a defaultdict? Actually, internally the co-occurance graph uses a subclass of `collections.defaultdict`. This subclass is a thin wrapper around the stdlib defaultdict, but it slides in a (user-specifiable transformation) between the objects that are fed to `__getitem__` and `__setitem__` and what is given to the underlying defaultdict. Think of it like some light-weight middleware.

You provide a _callable_ which accepts an object and converts that object to a string. How this conversion is done is up to you. The Co-Occurance graph simply applies this callable to whatever is given to get/setitem and feeds the output to the underlying defaultdict.

By default this callable is just the builtin `str` function. To customize this behaviour, provide a callable as the value to the `token_mapper_class` argument:

```python
def my_token_mapper_func(token: spacy.tokens.Token) -> bool:
    ...  # Transform the token to a string
    return token_as_string

r = Rake(token_mapper_class=my_token_mapper_func)
```

Why might you want to do this? Perhaps you want the tokens "Hello" and "hello" to be mapped to the same vertex in the co-occurance graph. This is not the default behaviour, but can be achieved by, say, the function:

```python
def f(token):
    return str(token).lower()

r = Rake(token_mapper_class=f)
```

Or perhaps you want `is` and `are` to be mapped to the same vertex (via the lemma `be`). This is not the default behaviour, but can be achieved by, say, the function:

```python
def g(token):
    return token if isinstance(str, token) else token.lemma_.lower()

r = Rake(token_mapper_class=g)
```

**Important**: The callable passed as `token_mapper_class` must be idempotent. Specifically, it should satisfy the property:

```python
token_mapper(t) == token_mapper(token_mapper(t))
```

To circle back, this how tokens (that are used to "index into" the co-occurance graph) and the graph itself play together nicely.

Token mapper classes included in this package can be found in the `rake_spacy.mappers` module.
