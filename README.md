Overview
========

This package contains a set of [Spacy](https://spacy.io/) components
which can be installed into a Spacy pipeline. They annotate the Spacy
parse tree with additional attributes that make it easy to summarize
information about features of student writing.

Before You Install
------------

It is helpful to note that the use of AWE Components is best tested using [AWE_Workbench](https://github.com/ETS-Next-Gen/AWE_Workbench), which utilizes the features defined in AWE Components. There are a series of automatic tests which can be run to verify or validate AWE Components; in addition, there are examples, a web server for parsing documents, and an interactive document highlighting tool for visualizing the document features which are derived from AWE Components.

See AWE Workbench's installations steps and verify that you'd want to use it instead of installing AWE Components directly.

Installation
------------

Set up Python 3.11. If you wish to use `conda`:

    conda create -n test_install python=3.11 pip
    pip install pip --upgrade
    conda activate test_install

If you wish to use plain old `pip` with `virtualenvwrapper`:

    mkvirtualenv awe_components --python=/usr/bin/python3.11
    pip install pip --upgrade

Then from the AWE Workbench Components directory:

    pip install .
    python -m awe_components.setup.data --install

Or for development:

    pip install -e .
    python -m awe_components.setup.data --develop

You MUST run the `setup.data` script before using the system.

This module is called by the AWE Workbench module, which requires
it. You would not normally run it by itself.

Pipelines defined by this system
--------------------------------

1. `lexicalfeatures` -- gives access to a broad range of open-source
   lexical features describing properies of English words
2. `syntaxdiscoursefeatures` -- gives access to a range of features to
   describe things like sentence complexity and cohesion
3. `viewpointfeatures` -- gives access to a range of features that help
   describe perspective/viewpoint, argumentation, and narrative.
4. `lexicalclusters` -- divides the content words of the document into
   clusters of associated words
5. `contentsegmentation` -- identifies rough approximations to argument
   text units, e.g., main ideas, supporting ideas, details

The `SpacyTextBlob` pipeline is also called by the AWE Workbench, and
is necessary to calculate some of the features created by the
`viewpointfeatures` and `lexicalfeatures` modules.

Overview of AWE Workbench
-------------------------

What is the AWE Workbench?

This package is an open-source natural language processing pipeline designed to support automated writing evaluation  (AWE) – automated scoring and feedback of student essays and other educational materials. As such, it is intended to identify text features that may be needed

* To evaluate the quality of student writing

* To visualize significant features of student or model texts

* To analyze text content

* To provide instructional feedback

The AWE Workbench draws upon several open-source packages widely used in the NLP community. These include LanguageTool, an open source grammar/spell checker, the Spacy parser, two Spacy modules (Coreferee, for coreference resolution, and Holmes Extractor, for content matching and topic analysis), an instance of the BERT deep learning transformer supplied by HuggingFace, the open source spell-correction modules PySymSpell and Neuspell, and various lexical resources including WordNet, the Wordfreq word frequency module, the Morpholex morphology database, several academic word lists, two different sentiment lexicons, and a number of custom lexicons. Details for the lexical resources included in the AWE Workbench are noted in the license file. The AWE Workbench integrates these resources and provides a number of additional facilities for NLP analysis of student texts, as outlined below.

The AWE Workbench is designed to support analysis of the linguistic features of texts. As such, it is similar to various existing software tools (see, for example, https://www.linguisticanalysistools.org/tools.html), but at the time of writing it is the only general, open-source AWE software package of which the authors are aware that is available under the GNU Affero General Public License, rather than more restrictive licenses like the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International license. Currently, the package only supports English, though many of the modules included in it are multilingual. The authors welcome participation from members of the open-source community who may wish to support further development of the AWE Workbench (in English, or by extending it to other languages). We hope that it will help researchers and developers implement high-quality, state-of-the-at automated writing evaluation tools and applications.

Overall Structure of the Package
--------------------------------

The AWE Workbench provides four basic services:

* A LanguageTool wrapper. LanguageTool identifies grammar, usage, mechanics, and style errors, and provides feedback text that can be displayed to the user. Our LanguageTool wrapper provides an additional layer of classification that makes it easier to identify errors that address the same specific construct, such as subject/verb agreement within grammar, or likely types within spelling.

* A spelling correction module that incorporates PySymSpell and Neuspell. This module is specifically designed to be used to standardize the spelling of student texts written on a common topic, so that they can be subjected to further NLP analysis. It works best when applied to a corpus of student texts that reflects a range of spelling abilities, so that it can infer the correct spellings of words commonly used to address a specific assignment. When this information is not available, or is not sufficient, it falls back on the spell-correction facilities provided by Neuspell, a state-of-the-art transformer-based spell-corrector.

* A wrapper for the BERT transformer that allows the user to extract the probability of words in context. BERT can, of course, be used independently to build a variety of classifiers, though currently the AWE Workbench uses it only in a few, relatively limited contexts.

* A natural language processing (NLP) pipeline built around the Spacy parser. In addition to the Coreferee and Holmes Extractor modules, this pipeline includes the following custom components:

* A lexical feature component, which calculates a variety of word-based features, including token, lemma and root counts and frequencies, size of word families, syllable counts, word length, latinate and academic vocabulary status,  number of word senses, and measures of animacy, abstractness/concreteness, sentiment and subjectivity.

* A lexical cluster component, which provides an agglomerative clustering of the words that appear within a document using Spacy word vectors. These clusters are used to support a number of other statistics, such as measures of the distribution of the largest clusters (which are likely to reflect the primary topic of the document) and of the difficulty of the vocabulary that appears in other, smaller clusters (which are likely to reflect development of secondary topics).

* A syntax and discourse feature component, which provides measures of the number and length of sentences and paragraphs, the number and types of transition words used to mark discourse segments, and the number and length of pronominal coreference chains; measures of syntactic complexity and variety such as depth of embedding and the number of tags and types of grammatical dependencies deployed in a text, and measures of textual coherence, such as the cosine similarity of content words in adjacent sentences or across paragraph boundaries and other discourse transitions.

* A viewpoint feature component, which identifies viewpoint predicates, such as emotion, cognition, and perception words, stance markers, which indicate the subjective perspective of the writer, and markers of direct and indirect speech. The viewpoint feature component uses this information to determine what parts of a text are to be evaluated as reflecting a specific viewpoint. The resulting features are used to support various genre-specific analyses, such as identification of the parts of a text that contain argumentation, or which contain references to the actions and mental states of story characters.

* A content segmentation component, which identifies major content blocks marked by chains of repeated or related words, and which determines whether individual sentences have significant content that address the main ideas of an essay and/or overlap with specified prompt language.

These modules are by design rule-based, rather than statistical in nature, and intended to capture features of the text that can be explicitly identified and labeled using linguistic knowledge. They capture dimensions that have been established as relevant to essay quality and structure in the research literature, which can be an important consideration when building educational applications. These criteria led to the exclusion of some forms of text analysis, such as rhetorical-structure parsing, which depend critically on a statistical model. However, the linguistic features supported by the AWE workbench include most of the surface cues that such models exploit. The outputs created by the AWE Workbench can easily be used as inputs to more sophisticated, statistical classifiers, but if used without modification, they are intended to provide a strong baseline for analyzing student texts.

It is important to note that while the features deployed in the AWE Workbench may bear a family resemblance to features deployed in commercial AWE systems, they were for the most part selected because they capture what patent law would recognize as prior art – well-known, long-established methods for analyzing student texts using natural language processing techniques. Places where the AWE Workbench contains novel contributions are identified below. Such contributions may be subject to patent applications filed by the authors, but are nonetheless released for use under the terms of the Gnu Affero public license.

Also note that we include a simple server API, to support use cases where the AWE Workbench needs to run in a distributed environemtn.

Contributions that Go Beyond Prior Art
--------------------------------------

While largely based on prior art, the AWE Workbench does contain several significant innovations, which are outlined below.

Open-Source Concreteness Estimates. The largest, most reliable databases of lexical concreteness estimates for English are not available under the GNU Affero license. The AWE Workbench applies an algorithm that generalizes the freely-available Glasgow concreteness norms to other English words, using the WordNet ontology. This algorithm was developed by the authors as part of IES grant R205A210297. We are currently validating these estimates. However, they appear to be accurate enough to support their use as an approximation to larger datasets of human concreteness ratings, such as those provided by Brysbaert, Warriner, & Kuperman, 2013.

The ETS Viewpoint Lexicon. The AWE Workbench defines an approach to subjectivity and stance-taking that depends on a fine-grained lexical classification of cognition, perception, emotion, and communication predicates, developed by the authors as part of IES grant R205A210297. Using this classification, the AWE Workbench uses syntactic dependencies identified by the Spacy parser to locate noun phrases and clauses that define the viewpoints attributed to these predicates and the propositions to which those viewpoints apply. This makes it possible to identify explicit argument language, important features of narrative, and other aspects of text structure that depend upon stance and viewpoint. As such, it covers similar ground to such widely-used resources as the MPQA argument lexicon, but in greater depth, and follows a design philosophy similar to that defended by St. Dizier (2020). We are currently validating the use of this lexicon. However, its results appear to be accurate enough to use to identify potential argument or narrative language in a text.

Applications
------------

Like most state-of-the-art AWE systems, the AWE Workbench can be used to extract summary features that can be used to predict human scores or classify student responses into meaningful categories. We are currently validating these uses with publicly available datasets. However, since equivalent features to those deployed by the AWE Workbench have already been validated elsewhere, the user can proceed with similar applications in advance of our completion of this effort. We will include a script for extracting summary features with the distribution we are preparing for public release.


