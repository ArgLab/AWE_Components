#!/usr/bin/env python3
# Copyright 2022, Educational Testing Service

import os
import srsly
import imp

from enum import Enum
from collections import OrderedDict
from spacy.tokens import Doc, Span, Token
from spacy.language import Language

from scipy.spatial.distance import cosine
# Standard cosine distance metric

from nltk.corpus import wordnet
# (a lot more, but that's what we're currently using it for)

from .utility_functions import *
from ..errors import *
from importlib import resources


def defineMorpholex():
    from awe_components.components.lexicalFeatures import morpholex


@Language.factory("viewpointfeatures", default_config={"fast": False})
def ViewpointFeatures(nlp, name, fast: bool):
    return ViewpointFeatureDef(fast)


class ViewpointFeatureDef:
    """
     This class handles viewpoint and perspective phenomena, including
     recognition of direct vs. indirect speech, finding the implied
     viewpoint of mental state verbs and verbs of saying, recognizing
     stance elements that imply a subjective judgment from a particular
     viewpont, and recognizing other elements that involve explicit
     argumentation.
    """

    # with resources.path('awe_lexica.json_data',
    #                     'stancePerspectiveVoc.json') as filepath:

    with resources.as_file(
        resources.files('awe_lexica').joinpath('json_data').joinpath('stancePerspectiveVoc.json')
    ) as filepath:
        STANCE_PERSPECTIVE_PATH = filepath

    # with resources.path('awe_lexica.json_data',
    #                     'morpholex.json') as filepath:

    with resources.as_file(
        resources.files('awe_lexica').joinpath('json_data').joinpath('morpholex.json')
    ) as filepath:
        MORPHOLEX_PATH = filepath

    stancePerspectiveVoc = {}
    morpholex = {}
    is_nominalization = {}

    calculatePerspective = True

    def package_check(self, lang):
        if not os.path.exists(self.STANCE_PERSPECTIVE_PATH):
            raise LexiconMissingError(
                "Trying to load AWE Workbench Syntaxa and Discourse Feature \
                 Module without supporting datafile {}".format(filepath)
            )
        if not os.path.exists(self.MORPHOLEX_PATH):
            raise LexiconMissingError(
                "Trying to load AWE Workbench Syntaxa and Discourse Feature \
                 Module without supporting datafile {}".format(filepath)
            )

    def load_lexicon(self, lang):
        """
         Load the lexicon that contains word class information we 
         will need to process perspective and argumentation elements.
         Identify nominalizations.
        """
        self.stancePerspectiveVoc = \
            srsly.read_json(self.STANCE_PERSPECTIVE_PATH)

        # would really rather not reimport, but
        # importing from lexicalFeatures is a bit complicated.
        # TBD: fix this
        self.morpholex = \
            srsly.read_json(self.MORPHOLEX_PATH)

        # Nominalizations are by definition not concrete

        for token in self.morpholex:
            morpholexsegm = \
                self.morpholex[token]['MorphoLexSegm']
            if morpholexsegm.endswith('>ship>') \
               or morpholexsegm.endswith('>hood>') \
               or morpholexsegm.endswith('>ness>') \
               or morpholexsegm.endswith('>less>') \
               or morpholexsegm.endswith('>ion>') \
               or morpholexsegm.endswith('>tion>') \
               or morpholexsegm.endswith('>ity>') \
               or morpholexsegm.endswith('>ty>') \
               or morpholexsegm.endswith('>cy>') \
               or morpholexsegm.endswith('>al>') \
               or morpholexsegm.endswith('>ance>'):
                self.is_nominalization[token] = True
        self.morpholex = None

    def markPerspectiveSpan(self, doc):
        ''' Utility function to mark point of view spans
            on the parse tree
        '''       
        if doc._.vwp_perspective_spans_ is not None:
            return

        # Find implied perspectives
        self.perspectiveMarker(doc)

        # Identify the perspectives
        # that apply to individual tokens in the text
        self.set_perspective_spans(doc)

    def __call__(self, doc):
        """
            Pass the document through unchanged. We will calculate
            (and if necessary, store) extended attribute values upon
            request
        """
        return doc

    def AWE_Info(self,
                 document: Doc,
                 infoType='Token',
                 indicator='pos_',
                 filters=[],
                 transformations=[],
                 summaryType=None):
        ''' This function provides a general-purpose API for
            obtaining information about indicators reported in
            the AWE Workbench Spacy parse tree.

            This is a general-purpose utility. Cloning inside
            the class to simplify the add_extensions class
        '''
        return AWE_Info(document, infoType, indicator, filters,
                        transformations, summaryType)

    def lexEval(self, token: Token, attribute: str):
        '''
            Utility function for processing ArgVoc lexicon data
            when we override the default behavior for loading
            stance/perspective vocabulary information into
            extended attributes we can reference
        '''
        pos = token.pos_
        lemma = token.lemma_
        if lemma in self.stancePerspectiveVoc:
            if pos in self.stancePerspectiveVoc[lemma]:
                if attribute in self.stancePerspectiveVoc[lemma][pos]:
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False

    def vwp_evaluation(self, token):
        '''
            Override definition for vwp_evaluation --
            using lexicon data if rules given here
            are not satisfied
       '''
        # rule based evaluation of certain syntactic
        # and lexical patterns as evaluative in nature
        if emphatic_adverb(token) \
           or emphatic_adjective(token) \
           or elliptical_verb(token) \
           or token.pos_ == 'INTJ' \
           or token._.vwp_probability \
           or token._.vwp_likelihood \
           or token._.vwp_evaluated_role:
            token._.vwp_evaluation_ = True

        # certain as attributive adjective
        if token._.vwp_evaluation_ is None \
           and token.dep_ == 'amod' \
           and token.lower_ == 'certain':
            token._.vwp_evaluation_ = False

        # modal have to
        if token._.vwp_evaluation_ is None \
           and token.lower_ in present_semimodals \
           and token.i+1 < len(token.doc) \
           and token.nbor().tag_ == 'TO':
            token._.vwp_evaluation_ = True

        # plan words with modal complements
        if token._.vwp_evaluation_ is None \
           and token.tag_ == 'TO' \
           and token.head.pos_ in ['NOUN', 'VERB', 'ADJ'] \
           and token.head.head._.vwp_plan:
            token._.vwp_evaluation_ = True

        # plan words asserted as a predicat
        if token._.vwp_evaluation_ is None \
           and token._.vwp_plan \
           and token.dep_ in object_predicate_dependencies:
            token.head._.vwp_evaluation_ = True

        # use override values if available
        if token._.vwp_evaluation_ is not None:
            return token._.vwp_evaluation_

        # otherwise use the normal stance/perspective lexicon
        if self.check_ngrams(token, 'evaluation'):
            return True
        return self.lexEval(token, 'evaluation')

    def vwp_likelihood(self, token):
        '''
           Override definition for vwp_likelihood --
           using lexicon data if rules given here
           are not satisfied
        '''
        if token._.vwp_likelihood_ is not None:
            return token._.vwp_likelihood_
        if self.check_ngrams(token, 'likelihood'):
            return True
        return self.lexEval(token, 'likelihood')

    def vwp_probability(self, token):
        '''
            Override definition for vwp_probability --
            using lexicon data if rules given here
            are not satisfied
        '''
        if token._.vwp_probability_ is not None:
            return token._.vwp_probability_
        if self.check_ngrams(token, 'probability'):
            return True
        return self.lexEval(token, 'probability')

    def vwp_argument(self, token):
        '''
           Override definition for vwp_argument --
           using lexicon data if rules given here
           are not satisfied
        '''
        if token._.transition \
           and '\n' not in token.text \
           and token._.transition_category != 'temporal':
            return True
        if token._.vwp_argument_ is not None:
            return token._.vwp_argument_
        if self.check_ngrams(token, 'argument'):
            return True
        return self.lexEval(token, 'argument')

    def has_governing_subject(self, token):
        if not token.doc._.has_governing_subject:
            for item in token.doc:
                if isRoot(item):
                    self.markImplicitSubject(item, token.doc)
            token.doc._.has_governing_subject = True
        return token._.has_governing_subject_

    def governing_subject(self, token):
        if not token.doc._.has_governing_subject:
            for item in token.doc:
                self.markImplicitSubject(item, token.doc)
            token.doc._.has_governing_subject = True
        return token._.governing_subject_

    def vwp_argumentation(self, token):
        self.markPerspectiveSpan(token.doc)
        if not token.doc._.vwp_argumentation:
            for item in token.doc:
                self.mark_argument_words(item, token.doc)
            token.doc._.vwp_argumentation = True
        return token._.vwp_argumentation_

    def vwp_perspective(self, token):
        self.markPerspectiveSpan(token.doc)
        return token._.vwp_perspective_

    def subjectiveWord(self, token):
        if token.text in ['?', '!'] \
           or token._.vwp_evaluation \
           or (token._.vwp_evaluated_role
               and token.pos_ == 'NOUN') \
           or ((token._.vwp_emotion
               or token._.vwp_character)
               and token._.has_governing_subject
               and token.doc[token._.governing_subject]._.animate) \
           or token._.vwp_hedge \
           or (token._.vwp_emotional_impact
               and (getLogicalObject(token) is None
                    or getLogicalObject(token)._.animate)) \
           or token._.vwp_possibility \
           or (token._.vwp_plan
               and token._.has_governing_subject
               and not in_past_tense_scope(token)) \
           or token._.vwp_future:
            return True
        return False

    def vwp_statements_of_fact(self, tokens):
        '''
           Identify sentences stated in entirely objective language
        '''
        return self.statements_of_fact_prep(tokens, lambda x: x==0)

    def vwp_statements_of_opinion(self, tokens):
        '''
           Identify sentences stated in entirely subjective language
        '''
        theList = self.statements_of_fact_prep(tokens, lambda x: x>0)
        opinionList = []
        for item in theList:
             newItem = item
             newItem['name'] = 'subjective'
             opinionList.append(newItem)
        return opinionList

    def statements_of_fact_prep(self, tokens, relation):
        '''
           Underlying function that identifies sentences
           with or without subjective stance markers in them
        '''
        factList = []
        j = 0
        nextRoot = 0
        currentHead = None
        for i in range(0, len(tokens)):
            # Once we find an objective element, we can stop
            # counting so we skip ahead to the head of the
            # next setence
            if i < nextRoot:
                continue

            # we don't count subjective elements in quotations
            if tokens[i]._.vwp_quoted:
                continue

            # tensed sentences containing not containing 
            # modal auxiliaries in the root clause
            # (And which aren't conjoined to a preceding
            #  clause)
            if isRoot(tokens[i]) \
               and not (tokens[i].dep_ == 'conj'
                        and tensed_clause(tokens[i].head)
                        and 'MD' in [child.tag_
                                     for child
                                     in tokens[i].head.children]):
                numSubjective = 0
                currentHead = tokens[i]

                # start counting subjective words we encounter
                # if viewpoint is left implicit
                if self.subjectiveWord(tokens[i]) \
                   and len(tokens[i]._.vwp_perspective) == 0:
                    numSubjective += 1

                # also count subjective words if viewpoint is
                # explicitly 1st or 2nd person
                if len(tokens[i]._.vwp_perspective) > 0 \
                   and (tokens[tokens[i]._.vwp_perspective[0]
                               ].lower_
                        in first_person_pronouns
                        or tokens[tokens[i]._.vwp_perspective[0]
                                  ].lower_
                        in second_person_pronouns) \
                   and (self.subjectiveWord(tokens[i])):
                    numSubjective += 1

                # Use of potential viewpoint predicates with first or second
                # person viewpoint governors in root position is a mark of
                # subjectivity
                if isRoot(tokens[i]) \
                   and not tokens[i]._.vwp_emotional_impact \
                   and (generalViewpointPredicate(tokens[i])
                        or tokens[i]._.vwp_emotion) \
                    and tokens[i]._.has_governing_subject \
                    and (tokens[tokens[i]._.governing_subject
                                ].lower_ in first_person_pronouns
                         or tokens[tokens[i]._.governing_subject
                                   ].lower_ in second_person_pronouns):
                    numSubjective += 1

                # Emotional impact words with a logical object
                # (it surprised me) count as subjective
                if isRoot(tokens[i]) \
                   and tokens[i]._.vwp_emotional_impact \
                   and getLogicalObject(tokens[i]) is not None \
                   and (getLogicalObject(tokens[i]).lower_
                        in first_person_pronouns
                        or getLogicalObject(tokens[i]).lower_
                        in second_person_pronouns):
                    numSubjective += 1

                # Special case: imperatives are implicitly subjective
                # even though there is no explicit evaluation word
                if tokens[i] == self.getHeadDomain(tokens[i]) \
                   and tokens[i].lower_ == tokens[i].lemma_ \
                   and not in_past_tense_scope(tokens[i]) \
                   and 'expl' not in [left.dep_ for left in tokens[i].lefts] \
                   and getSubject(tokens[i]) is None:
                    numSubjective += 1

                # Scan through the subtree for the current token.
                for child in tokens[i].subtree:
                
                    # skip the current token
                    if child == tokens[i]:
                        continue

                    # skip quoted text
                    if child._.vwp_quoted:
                        continue

                    # when we encounter the root of
                    # a tensed clause that isn't an adverbial
                    # modifier or a subjectless predicate,
                    # skip the focus of search forward to this word
                    # and break out of our search through the subtree
                    if isRoot(child) \
                       and child.dep_ in ['ROOT', 'conj'] \
                       and child != tokens[i] \
                       and tensed_clause(child) \
                       and getSubject(child) is not None \
                       and not (child.dep_ == 'conj'
                                and 'SCONJ' in [gchild.pos_
                                                for gchild
                                                in tokens[i].children]) \
                       and not (child.dep_ == 'conj'
                                and tensed_clause(child.head)
                                and 'MD' in [gchild.tag_
                                             for gchild
                                             in child.head.children]):
                        currentHead = child
                        break

                    # if a child inside the current clause is subjective,
                    # and viewpoint is implicit, increase the count
                    if self.subjectiveWord(child) \
                       and len(child._.vwp_perspective) == 0:
                        numSubjective += 1
                        break

                    # if a child inside the current clause is subjective
                    # and viewpoint is explicitly 1st or 2nd person,
                    # increase the count
                    if len(child._.vwp_perspective) > 0 \
                       and (tokens[child._.vwp_perspective[0]].lower_
                            in first_person_pronouns
                            or tokens[child._.vwp_perspective[0]].lower_
                            in second_person_pronouns) \
                       and self.subjectiveWord(child):
                        numSubjective += 1
                        break

                    # Special case: cognitive, communication or
                    # argument predicates as subjects of the
                    # domain head are implicitly subjective
                    if child.dep_ in subject_dependencies \
                       and (len(child._.vwp_perspective) == 0
                            or (tokens[child._.vwp_perspective[0]
                                       ].lower_
                                in first_person_pronouns)
                            or (tokens[child._.vwp_perspective[0]
                                       ].lower_
                                in second_person_pronouns)) \
                       and not child._.vwp_sourcetext \
                       and (generalViewpointPredicate(child)
                            or child._.vwp_emotion):
                        numSubjective += 1

                    # possibility words (mostly modals) that
                    # are at the head of the current viewpoint
                    # domain are implicitly subjective
                    if child._.vwp_possibility \
                       and child == self.getHeadDomain(child):
                        numSubjective += 1
                        break
                        
                # if we found/did not find any subjective words,
                # depending on our flag settings, add the
                # current sentence to our list of spans and
                # move on.
                if relation(numSubjective):
                    start, end = rootTree(currentHead,
                                          currentHead.i,
                                          currentHead.i)
                    entry = \
                        newSpanEntry('objective',
                                     start,
                                     end,
                                     tokens,
                                     tokens[end].idx \
                                     + len(tokens[end].text_with_ws) \
                                     - tokens[start].idx)
                    factList.append(entry)
                    nextRoot = end + 1
                    numSubjective = 0
        return factList

    def vwp_emotionword(self, token):
        ''' Return list of emotion words for all tokens in document
        '''
        if token._.vwp_emotion \
           or token._.vwp_emotional_impact:
            return True
        else:
            return False

    def vwp_argumentword(self, token):
        ''' Return list of token indexes for words that meet the standard
            for being explicit argumentation words
        '''
        if token._.vwp_argument \
           or token._.vwp_certainty \
           or token._.vwp_necessity \
           or token._.vwp_probability \
           or token._.vwp_likelihood \
           or token._.vwp_surprise \
           or token._.vwp_qualification \
           or token._.vwp_emphasis \
           or token._.vwp_accuracy \
           or token._.vwp_information \
           or token._.vwp_relevance \
           or token._.vwp_persuasiveness \
           or token._.vwp_reservation \
           or token._.vwp_qualification \
           or token._.vwp_generalization \
           or token._.vwp_illocutionary \
           or token._.vwp_argue:
            return token.i

    def vwp_explicit_argument(self, token):
        ''' Return list of indexes to words that belong to argumentation
            sequences and can be classified as subjective/argument language
            This is a more restrictive definition in some ways than
            vwp_argumentword, since the viewpoint relationships have to
            be properly subjective.
        '''
        #return [token.i for token in tokens
        if token._.vwp_argumentation \
           and (token._.vwp_argument
                or token._.vwp_certainty
                or token._.vwp_necessity
                or token._.vwp_probability
                or token._.vwp_likelihood
                or token._.vwp_surprise
                or token._.vwp_risk
                or token._.vwp_cause
                or token._.vwp_tough
                or token._.vwp_qualification
                or token._.vwp_emphasis
                or token._.vwp_accuracy
                or token._.vwp_information
                or token._.vwp_relevance
                or token._.vwp_persuasiveness
                or token._.vwp_reservation
                or token._.vwp_qualification
                or token._.vwp_generalization
                or token._.vwp_illocutionary
                or token._.vwp_manner
                or token._.vwp_importance
                or ((token._.vwp_say
                     or token._.vwp_interpret
                     or token._.vwp_perceive
                     or token._.vwp_think
                     or token._.vwp_argue
                     or (token._.vwp_plan
                         and not token._.vwp_physical
                         and not token._.vwp_social))
                    and (not token.pos_ == 'VERB'
                         or (not token._.has_governing_subject
                             and tensed_clause(token))
                         or (token._.has_governing_subject
                             and (token.doc[
                                  token._.governing_subject
                                  ]._.animate
                                  or token.doc[
                                     token._.governing_subject
                                     ].lemma_ in inanimate_3sg_pronouns
                                  or token.doc[
                                     token._.governing_subject
                                     ]._.vwp_cognitive
                                  or token.doc[
                                     token._.governing_subject
                                     ]._.vwp_plan
                                  or token.doc[
                                     token._.governing_subject
                                     ]._.vwp_abstract
                                  or token.doc[
                                     token._.governing_subject
                                     ]._.vwp_information
                                  or token.doc[
                                     token._.governing_subject
                                     ]._.vwp_possession
                                  or token.doc[
                                     token._.governing_subject
                                     ]._.vwp_relation
                                  or token.doc[
                                     token._.governing_subject
                                     ]._.vwp_communication))))):
            return True

    # This set of attributes are dependent upon having called the
    # vwp_direct_speech function which calculates direct speech spans.
    # So we record the flag information on attributes with an extra
    # underscore on them, and then define the function without the
    # underscore in such a way that it makes sure vwp_direct_speech has
    # been called before we check the underscore attribute. This
    # enables us to call all of the direct speech attributes in a 
    # 'just in time' fashion w/o having to do the relatively costly
    # vwp_direct_speech calculations as part of the main parse.

    def vwp_direct_speech_verb(self, token):
          # make sure direct speech info has been set
          self.vwp_direct_speech(token.doc)
          return token._.vwp_direct_speech_verb_

    def vwp_in_direct_speech(self, token):
          # make sure direct speech info has been set
          self.vwp_direct_speech(token.doc)
          return token._.vwp_in_direct_speech_

    def vwp_speaker(self, token):
          # make sure direct speech info has been set
          self.vwp_direct_speech(token.doc)
          return token._.vwp_speaker_

    def vwp_speaker_refs(self, token):
          # make sure direct speech info has been set
          self.vwp_direct_speech(token.doc)
          return token._.vwp_speaker_refs_

    def vwp_addressee(self, token):
          # make sure direct speech info has been set
          self.vwp_direct_speech(token.doc)
          return token._.vwp_addressee_

    def vwp_addressee_refs(self, token):
          # make sure direct speech info has been set
          self.vwp_direct_speech(token.doc)
          return token._.vwp_addressee_refs_

    def vwp_abstract(self, token):
        if (token._.concreteness is not None
            and token._.concreteness < 3.5) \
           or token._.abstract_trait \
           or token.lower_ in self.is_nominalization:
            return True
        else:
            return False

    def vwp_sentiment(self, token):
        if token.doc._.negation_tokens is None:
            # Mark sentiment properly under scope of negation, using
            # sentiWord weights as base lexical sentiment polarities
            negation_tokens = self.propagateNegation(token.doc)
            for token in token.doc:
                if isRoot(token):
                    self.traverseTree(token, negation_tokens)
            token.doc._.negation_tokens = negation_tokens
        return token._.vwp_sentiment_

    def vwp_tone(self, token):
        if token.doc._.negation_tokens is None:
            # Mark sentiment properly under scope of negation, using
            # sentiWord weights as base lexical sentiment polarities
            negation_tokens = self.propagateNegation(token.doc)
            for token in token.doc:
                if isRoot(token):
                    self.traverseTree(token, negation_tokens)
            token.doc._.negation_tokens = negation_tokens
        return token._.vwp_tone_

    def check_ngrams(self, token, attribute):
        '''
           Supporting function that marks words that are
           part of a multiword entry in the stance perspective
           dictionary with the correct attribute info.
        '''
        if token.i+1 < len(token.doc):
            bigram = token.lower_ + '_' \
                + token.doc[token.i+1].lower_
            if bigram in self.stancePerspectiveVoc:
                for pos in self.stancePerspectiveVoc[bigram]:
                    if attribute in self.stancePerspectiveVoc[bigram][pos]:
                        return True

            if token.i + 2 < len(token.doc):
                trigram = token.lower_ + '_' \
                    + token.doc[token.i+1].lower_  + '_' \
                    + token.doc[token.i+2].lower_
                if trigram in self.stancePerspectiveVoc:
                    for pos in self.stancePerspectiveVoc[trigram]:
                        if attribute in self.stancePerspectiveVoc[trigram][pos]:
                            return True

        if token.i > 0:
            
            bigram = token.doc[token.i - 1].lower_ \
                + '_' + token.lower_ 
            if bigram in self.stancePerspectiveVoc:
                for pos in self.stancePerspectiveVoc[bigram]:
                    if attribute in self.stancePerspectiveVoc[bigram][pos]:
                        return True
            if token.i+1 < len(token.doc):
                trigram = token.doc[token.i - 1].lower_ + '_' \
                    + token.lower_ + '_' \
                    + token.doc[token.i + 1].lower_ 
                if trigram in self.stancePerspectiveVoc:
                    for pos in self.stancePerspectiveVoc[trigram]:
                        if attribute in self.stancePerspectiveVoc[trigram][pos]:
                            return True

        if token.i > 1:
            trigram = token.doc[token.i - 2].lower_ + '_' \
                + token.doc[token.i - 1].lower_ + '_' \
                + token.lower_
            if trigram in self.stancePerspectiveVoc:
                for pos in self.stancePerspectiveVoc[trigram]:
                    if attribute in self.stancePerspectiveVoc[trigram][pos]:
                        return True

        return False

    def add_extensions(self):
        """
         Funcion to add extensions with getter functions that allow us
         to access the various viewpoint/argumentation functions built
         into viewpointFeatures
        """
        method_extensions = [self.AWE_Info]      
        docspan_extensions = [self.vwp_perspective_spans,
                              self.vwp_stance_markers,
                              self.vwp_direct_speech,
                              self.vwp_statements_of_fact,
                              self.vwp_social_awareness,
                              self.vwp_propositional_attitudes,
                              self.vwp_emotion_states,
                              self.vwp_character_traits,
                              self.vwp_statements_of_opinion,
                              self.vwp_egocentric,
                              self.vwp_allocentric,
                              self.vwp_interactive,
                              self.tense_changes,
                              self.concrete_details,
                              self.nominalReferences]
        token_extensions = [self.vwp_evaluation,
                            self.vwp_argument,
                            self.has_governing_subject,
                            self.governing_subject,
                            self.vwp_perspective,
                            self.vwp_abstract,
                            self.vwp_direct_speech_verb,
                            self.vwp_in_direct_speech,
                            self.vwp_speaker,
                            self.vwp_speaker_refs,
                            self.vwp_addressee,
                            self.vwp_addressee_refs,
                            self.vwp_cite,
                            self.vwp_attribution,
                            self.vwp_source,
                            self.vwp_argumentation,
                            self.vwp_explicit_argument,
                            self.vwp_emotionword,
                            self.vwp_sentiment,
                            self.vwp_tone,
                            self.vwp_claim,
                            self.vwp_discussion,
                            self.vwp_argumentword]
        setExtensionFunctions(method_extensions, 
                              docspan_extensions,
                              token_extensions)

        ##################################################
        # Register extensions for all the categories in  #
        # the stance lexicon                             #
        ##################################################
        vwp_extensions = []
        # find the list of all extensions in the dictionary
        for entry in self.stancePerspectiveVoc:
            for pos in self.stancePerspectiveVoc[entry]:
                for attribute in self.stancePerspectiveVoc[entry][pos]:
                    if attribute not in vwp_extensions:
                        vwp_extensions.append(attribute)

        # create getter functions for each attribute marked
        # in the stance perspective vocabulary and
        # create the corresponding extended attributes
        for attribute in vwp_extensions:
            def makeExtension(attribute):
                def getterFunc(token: Token):
                    pos = token.pos_
                    lemma = token.lemma_
                    if lemma in self.stancePerspectiveVoc:
                        if pos in self.stancePerspectiveVoc[lemma]:
                            if attribute in self.stancePerspectiveVoc[lemma][pos]:
                                return True
                    if self.check_ngrams(token, attribute):
                        return True
                    return False
                return getterFunc
            getterFunc = makeExtension(attribute)
            if not Token.has_extension('vwp_' + attribute):
                Token.set_extension('vwp_' + attribute, getter=getterFunc)

        #########################################################
        # Bookkeeping extensions to store data we don't want to #
        # recalculate once it has been calculated once.         #
        #########################################################

        # Index to the word that identifies the perspective that applies
        # to this token
        if not Token.has_extension('vwp_perspective_'):
            Token.set_extension('vwp_perspective_', default=None)

        # Special index that tracks the perspective of root words for
        # whole sentences. Default is empty list, implicitly the speaker.
        # Only filled when whole-clause viewpoint is signaled explicitly
        if not Token.has_extension('head_perspective'):
            Token.set_extension('head_perspective', default=[])

        # Next section:
        # Storage attributes for data we have to save off after we
        # calculate them.

        # markers for attributions
        if not Token.has_extension('vwp_attribution_'):
            Token.set_extension('vwp_attribution_', default=False)

        # markers for the source text or person indicated
        # by an attribution
        if not Token.has_extension('vwp_source_'):
            Token.set_extension('vwp_source_', default=False)

        # markers for citations
        if not Token.has_extension('vwp_cite_'):
            Token.set_extension('vwp_cite_', default=False)

        # markers for clauses that express propositional attitudes
        # on the part of a viewpoint controller
        if not Doc.has_extension('propositional_attitudes_'):
            Doc.set_extension('propositional_attitudes_',
                default=None, force=True)

        # markers for items identified as claims
        if not Token.has_extension('vwp_claim_'):
            Token.set_extension('vwp_claim_',
                default=False, force=True)

        # markers for indirect discussion of claims via
        # nominalization
        if not Token.has_extension('vwp_discussion_'):
            Token.set_extension('vwp_discussion_',
                                default=False,
                                force=True)

        # evaluation words (stance markers)
        if not Token.has_extension('vwp_evaluation_'):
            Token.set_extension('vwp_evaluation_',
                default=None, force=True)

        # override data for vwp_likelihood stance perspective dictionary
        if not Token.has_extension('vwp_likelihood_'):
            Token.set_extension('vwp_likelihood_',
                default=None, force=True)

        # override data for vwp_probability stance perspective dictionary
        if not Token.has_extension('vwp_probability_'):
            Token.set_extension('vwp_probability_',
                default=None, force=True)

        # override data for vwp_argument stance perspective dictionary
        if not Token.has_extension('vwp_argument_'):
            Token.set_extension('vwp_argument_',
                default=None, force=True)

        # Mapping of tokens to viewpoints for the whole document
        #
        # Our code creates separate lists by the viewpoint that
        # applies to each token: implicit first person,
        # explicit first person, explicit third person, and
        # for explicit third person, by the offset for the referent
        # that takes that perspective.
        Span.set_extension('vwp_perspective_spans_',
                           default=None,
                           force=True)
        Doc.set_extension('vwp_perspective_spans_',
                          default=None,
                          force=True)

        # Identification of stance markers that make a text more
        # subjective/opinion-based and less objective
        #
        # Our code creates separate lists by the viewpoint that
        # applies to the stance marker: implicit first person,
        # explicit first person, explicit third person, and
        # for explicit third person, by the offset for the referent
        # that takes that perspective.
        Doc.set_extension('vwp_stance_markers_',
                          default=None,
                          force=True)
        Span.set_extension('vwp_stance_markers_',
                           default=None,
                           force=True)

        # Marking of supporint words, stance markers,
        # argument words in sentences with explicit
        # argumentation in them.
        Doc.set_extension('vwp_argumentation',
                          default=False,
                          force=True)

        # SentiWord sentiment polarity ratings #
        if not Doc.has_extension('negation_tokens'):
            Doc.set_extension('negation_tokens', default=None)

        # Rating of positive or negative sentiment
        if not Token.has_extension('vwp_sentiment_'):
            Token.set_extension('vwp_sentiment_',
                                default=None)

        # Rating of positive or negative sentiment
        if not Token.has_extension('vwp_tone_'):
            Token.set_extension('vwp_tone_',
                                default=None)

        ##########################
        # Argumentative style    #
        ##########################

        if not Token.has_extension('vwp_argumentation_'):
            Token.set_extension('vwp_argumentation_', default=False)

        #######################################
        # Storage functions for direct speech #
        #######################################

        # List of spans that count as direct speech
        if not Doc.has_extension('direct_speech_spans'):
            Doc.set_extension('direct_speech_spans', default=None)

        # Flag that says whether a verb of saying (thinking, etc.)
        # is being used as direct speech ('John is happy, I think')
        # rather than as direct speech ('I think that John is happy.')
        if not Token.has_extension('vwp_direct_speech_verb_'):
            Token.set_extension('vwp_direct_speech_verb_',
                                default=False)

        if not Token.has_extension('vwp_in_direct_speech_'):
            Token.set_extension('vwp_in_direct_speech_',
                                default=False)

        # Flag identifying a nominal that identifies the
        # speaker referenced as 'I/me/my/mine' within a
        # particular stretch of direct speech
        if not Token.has_extension('vwp_speaker_'):
            Token.set_extension('vwp_speaker_', default=None)

        # List of all tokens (nominals or first person pronouns)
        # that refer to a speaker defined within a particular
        # stretch of direct speech
        if not Token.has_extension('vwp_speaker_refs_'):
            Token.set_extension('vwp_speaker_refs_', default=None)

        # Flag identifying a nominal (if present) that identifies
        # the speaker referenced as 'you/your/yours' within a
        # particular stretch of direct speech.
        if not Token.has_extension('vwp_addressee_'):
            Token.set_extension('vwp_addressee_', default=None)

        # List of all tokens (nominals or first person pronouns)
        # that refer to an addressee defined within a particular
        # stretch of direct speech
        if not Token.has_extension('vwp_addressee_refs_'):
            Token.set_extension('vwp_addressee_refs_', default=None)

        ##########################################################
        # Helper extensions for tracking viewpoint domains over  #
        # sentential scope                                       #
        ##########################################################

        # Flag that identifies whether a governing subjects have been
        # for the whole document.
        if not Doc.has_extension('has_governing_subject'):
            Doc.set_extension('has_governing_subject', default=False)

        # Flag that identifies whether a token has a governing subject.
        if not Token.has_extension('has_governing_subject_'):
            Token.set_extension('has_governing_subject_', default=False)

        # Pointer to a token's governing subject.
        if not Token.has_extension('governing_subject_'):
            Token.set_extension('governing_subject_', default=None)

        # A token's WordNet usage domain.
        if not Token.has_extension('usage'):
            Token.set_extension('usage', default=None)

    def __init__(self, fast: bool, lang="en"):
        super().__init__()
        self.package_check(lang)
        self.load_lexicon(lang)
        self.calculatePerspective = not fast
        vwp_extensions = []
        self.add_extensions()

    def synsets(self, token):
        return wordnet.synsets(token.orth_)

    def viewpointPredicate(self, token):
        """
         Viewpoint predicates cover a range of concepts, referenced
         using our spacy extensions. vwp_communication covers verbs
         of saying and similar elements in other parts of speech
         vwp_cognitive covers various sorts of mental state predicates.
         vwp_emotion covers emotion predicates where the experiencer
         is syntactically a subject. vwp_emotional_impact covers
         emotion predicates where the experiencer is syntactically
         an object. vwp_plan covers abstract predicates describing
         mental planning, strategy, and execution. vwp_argument covers
         predicates focusing on describing and evaluating argumentation.
        """

        if generalViewpointPredicate(token) \
           or token._.vwp_emotion \
           or token._.vwp_emotional_impact:
            return True
        return False

    def potentialViewpoint(self, token):
        """
         Potential viewpoint expressions are nominals that are either
         (a) animate, (b) in the 'vwp_sourcetext' or 'vwp_communication'
         categories that we loaded from our lexicon.
        """
        if token is not None and (token._.animate or token._.vwp_sourcetext) \
           and not token._.location:
            return True
        return False

    def find_abstract_trait_object(self, tok: Token):
        subj = None
        if 'of' in [child.lemma_ for child in tok.subtree] \
           and (self.viewpointPredicate(tok) or tok._.abstract_trait):
            for child in tok.subtree:
                if child.lemma_ == 'of':
                    subj = getPrepObject(child, ['of'])
                    break
        return subj

    def findViewpoint(self, predicate, tok: Token, lastDep, lastTok, hdoc):
        """
         Locate the explicit or implicit subjects of viewpoint-controlling
         predicates and mark the chain of nodes from the predicate to the
         controlling animate nominal as being within that nominal's viewpoint
         scope.
        """

        subj = None

        # Dative constructions like 'obvious to me' or 'give me
        # the belief' express viewpoint. The dative NP ('me') is
        # the viewpoint of the predicate ('obvious', 'belief')
        if (tok._.vwp_evaluation
            and 'IN' in [child.tag
                         for child in tok.children]) \
           or tok.head._.vwp_ditransitive:
            subj = getDative(tok)

        # Agent phrases of passive sentences can carry viewpoint
        elif 'agent' in [child.dep_ for child in tok.children]:
            for child in tok.children:
                if child.dep_ == 'agent':
                    subj = getPrepObject(child, ['by'])
                    break

        # Control structures can carry viewpoint
        elif (tok.dep_ in ['acl', 'acomp']
              and tok.head._.governing_subject is not None):
            subj = hdoc[tok.head._.governing_subject]
            return subj

        # But more generally it's the subject (or possessive of a noun)
        # that takes viewpoint for viewpint-taking predicates
        if tok._.vwp_emotional_impact:
            subj = getLogicalObject(tok)
            if subj is not None:
                return subj
            else:
                subj = getSubject(tok)

        if subj is None:
            subj = self.find_abstract_trait_object(tok)

        if subj is not None:

            # Recursively find subject of NPs headed by potential
            # viewpoint predicate nouns. E.g., 'Jenna's article
            # stated that this was a  problem.' -> Jenna is viewpoint
            # for 'stated'
            if (self.viewpointPredicate(subj)
                or tok._.abstract_trait) \
               and not subj._.animate \
               and subj != tok \
               and subj != lastTok:
                subj2 = self.findViewpoint(predicate,
                                           subj,
                                           subj.dep_,
                                           tok,
                                           hdoc)
                if subj2 is not None:
                    return subj2
                else:
                    return subj
            else:
                return subj

        # In general, subjectless constructions allow viewpoint to spread
        # to subjacent elements from animate nouns in viewpoint-taking roles
        # like subject (Jenny is starting to believe -> 'Jenny' is viewpoint,
        # same as in 'Jenny believes'.)
        elif (tok.dep_ in general_complements_and_modifiers
              and lastDep != 'ccomp'
              and lastDep != 'csubj'
              and lastDep != 'csubjpass'):
            # Note on restrictions above:
            # Complement clauses (comp, csubj, csubjpass) are opaque
            # to viewpoint from higher in the parse tree. (In: John
            # believes that Sue disagrees with him, 'disagree' must
            # be from Sue's POV, not John's)

            if tok._.vwp_emotional_impact \
               and tok.head._.vwp_raising:
                # Tough-movement verbs but not raising verbs are
                # transparent to viewpoint for emotional impact
                # verbs like 'alarm': 'Max is easy to please'
                # is from Max's viewpoint (Max is pleased) but
                # 'Max is certain to please' is default viewpoint
                # (the speaker's), with an unspecified object
                # viewpoint of who is pleased.
                return None
            if tok.head != tok.head.head:
                subj = self.findViewpoint(predicate,
                                          tok.head,
                                          tok.dep_,
                                          tok,
                                          hdoc)
            return subj

        # Allow light verbs to include their objects in the scope
        # of subject viewpoint (Maria has the right to argue that
        # I am wrong -> 'that I am wrong' is from Maria's POV)
        elif (tok.dep_ in underlying_object_dependencies
              and (tok.head._.vwp_cause
                   or tok.head._.vwp_possession
                   or tok.lemma_ in getLightVerbs())):
            subj = getSubject(tok)
            if subj is not None:
                if self.viewpointPredicate(subj) \
                   and not self.potentialViewpoint(subj):
                    subj = self.findViewpoint(predicate,
                                              subj,
                                              tok.dep_,
                                              tok,
                                              hdoc)
                if subj is not None:
                    return subj
                else:
                    return subj
            else:
                subj = self.findViewpoint(predicate,
                                          tok.head,
                                          tok.dep_,
                                          tok,
                                          hdoc)
                return subj
        else:
            return subj

    def registerViewpoint(self, hdoc, token, subj):
        """
         Utility function that resolves antecedents of pronouns
         and then sets the vwp_perspective_ extension attribute
         for a token to point to the right referent.
        """

        # really just going for consistency here. If it's
        # explicit 1st person perspective, we want to mark
        # everything in the domain as explicit rather than
        # implicit first person perspective
        if token._.governing_subject is not None \
           and hdoc[token._.governing_subject
                    ].lower_ in first_person_pronouns:
            token._.head_perspective = \
                [token._.governing_subject]

        subj._.vwp_perspective_ = [subj.i]
        token._.vwp_perspective_ = [subj.i]

    def markAttribution(self, tok, hdoc):
        if tok.dep_ == 'pobj' \
           and tok.head.lemma_ in dative_preps \
           and tok.i < self.getHeadDomain(tok).i \
           and tok._.vwp_perspective is not None \
           and tok.i in tok._.vwp_perspective:
            tok._.vwp_source_ = True
            tok._.vwp_attribution_ = True
            tok.head._.vwp_attribution_ = True
            if tok.head.head.lemma_ == 'accord':
                tok.head.head._.vwp_attribution_ = True        
        if (tok.dep_ == 'nsubj'
            and tok.head.dep_ == 'conj'
            and tok.head.head._.vwp_attribution_
            and generalArgumentPredicate(tok.head)) \
            or (tok.head._.governing_subject is not None
                and tok.dep_ in ['ccomp', 'csubjpass', 'acl']
                and tensed_clause(tok)
                and ((self.getHeadDomain(tok.head).dep_ is None
                     or isRoot(self.getHeadDomain(tok.head)))
                     or isRoot(self.getHeadDomain(tok.head).head)
                     or tok.head.dep_ == 'conj')
                and generalArgumentPredicate(tok.head)) \
            and not tok.left_edge.tag_.startswith('W') \
            and (hdoc[tok.head._.governing_subject].lower_
                 not in personal_or_indefinite_pronoun) \
            and (hdoc[tok.head._.governing_subject]._.animate
                 or (hdoc[tok.head._.governing_subject].lower_
                     in ['they',
                         'them',
                         'some',
                         'others',
                         'many',
                         'few'])
                 or hdoc[tok.head._.governing_subject].tag_ == 'PROPN'
                 or hdoc[tok.head._.governing_subject].tag_ == 'DET'
                 or hdoc[tok.head._.governing_subject]._.vwp_sourcetext):
            hdoc[tok.head._.governing_subject]._.vwp_source_ = True
            for child in hdoc[tok.head._.governing_subject].subtree:
                if child.i >= tok.head.i:
                    break
                child._.vwp_source_ = True
            tok.head._.vwp_attribution_ = True
            for child in tok.head.subtree:
                if child.i >= tok.head.i:
                    break
                child._.vwp_attribution_ = True

    def markCitedText(self, tok, hdoc):
        if tok.tag_ == '-LRB-' \
           and (tok.head.tag_ == 'NNP'
                or tok.head.ent_type_ in animate_ent_type
                or tok.head._.vwp_quoted):
            i = tok.i
            while (i < len(hdoc)
                   and hdoc[i].tag_ != '-RRB-'):
                hdoc[i]._.vwp_cite_ = True
                for child in hdoc[i].subtree:
                    child._.vwp_cite_ = True
                    tok._.vwp_cite_ = True
                i += 1
        elif tok.tag_ == '-LRB-':
            for child in tok.head.children:
                if child.tag_ == '-RRB-':
                    for i in range(tok.i + 1, child.i):
                        if hdoc[i].tag_ == 'NNP' \
                           or hdoc[i].ent_type_ in animate_ent_type \
                           or hdoc[i]._.vwp_quoted:
                            hdoc[i]._.vwp_cite_ = True
                            tok._.vwp_cite_ = True
                            for grandchild in hdoc[i].subtree:
                                grandchild._.vwp_cite_ = True
                            break

    def markAddresseeRefs(self, target, tok, addressee_refs):
        # Mark the addressee for the local domain,
        # which will be the object of the preposition
        # 'to' or the direct object
        if isRoot(target):
            target._.vwp_addressee_ = []
        for child2 in target.children:
            if child2.dep_ == 'dative' and child2._.animate:
                target._.vwp_addressee_ = [child2.i]
                tok._.vwp_addressee_ = [child2.i]
                if child2.i not in addressee_refs:
                    addressee_refs.append(child2.i)
                break
            if child2.dep_ == 'dobj' and child2._.animate:
                target._.vwp_addressee_ = [child2.i]
                tok._.vwp_addressee_ = [child2.i]
                if child2.i not in addressee_refs:
                    addressee_refs.append(child2.i)
                break
            if target._.vwp_addressee_ is not None:
                dativeNoun = getPrepObject(target, dative_preps)
                if dativeNoun is not None \
                   and dativeNoun._.animate:
                    target._.vwp_addressee_ = [dativeNoun.i]
                    dativeNoun._.vwp_addressee_ = [dativeNoun.i]
                    if child2.i not in addressee_refs:
                        addressee_refs.append(child2.i)
                        break
                dativeNoun = getPrepObject(target, ['at'])
                if dativeNoun is not None \
                   and dativeNoun._.animate:
                    target._.vwp_addressee_ = [dativeNoun.i]
                    dativeNoun._.vwp_addressee_ = [dativeNoun.i]
                    if child2.i not in addressee_refs:
                        addressee_refs.append(child2.i)
                    break
        return addressee_refs

    def vwp_direct_speech(self, hdoc):
        """
         Scan through the document and find verbs that control
         complement clauses AND contain an immediately dpeendent
         punctuation mark (which is the cue that this is some form
         of direct rather than indirect speech. E.g., 'John said,
         I am happy.'
        """
        
        if hdoc._.direct_speech_spans is not None:
            return hdoc._.direct_speech_spans

        lastRoot = None
        currentRoot = None
        speaker_refs = []
        addressee_refs = []
        for tok in hdoc:

            self.markCitedText(tok, hdoc)
            self.markAttribution(tok, hdoc)

            if '\n' in tok.text:
                speaker_refs = []
                addressee_refs = []
                lastRoot = currentRoot
                currentRoot = tok
            else:
                if currentRoot != getRoot(tok):
                    lastRoot = currentRoot
                currentRoot = getRoot(tok)

            # Special case: quote introduced by tag word
            # at end of previous sentence
            target = tok.head
            if tok == currentRoot:
                left = currentRoot.left_edge
                start = currentRoot.left_edge
                while (left.i > 0
                       and (left.tag_ == '_SP'
                            or left.dep_ == 'punct')):
                    left = left.nbor(-1)
                while (start.i + 1 < len(hdoc)
                       and start.tag_ == '_SP'):
                    start = start.nbor()
                target = left

            # If we match the special case with the taq word
            # being a verb or speaking, or else match the
            # general case where the complement of a verb of
            # speaking is a quote, then we mark direct speech
            if (tok == currentRoot
                and quotationMark(start)
                and not left._.vwp_plan
                and not left.head._.vwp_plan
                and (generalViewpointPredicate(left)
                     or generalViewpointPredicate(left.head))) \
                or (tok.dep_ in ['ccomp',
                                 'csubjpass',
                                 'acl',
                                 'xcomp',
                                 'intj',
                                 'nsubj']
                    and tok.head.pos_ in content_pos
                    and not tok.head._.vwp_quoted
                    and not tok.head._.vwp_plan
                    and generalViewpointPredicate(tok.head)):

                context = [child for child in target.children]
                if len(context) > 0 and context[0].i > 0:
                    leftnbor = context[0].nbor(-1)
                    if quotationMark(leftnbor):
                        context.append(leftnbor)
                if len(context) > 0 and context[0].i + 1 < len(hdoc):
                    rightnbor = context[0].nbor()
                    if quotationMark(rightnbor):
                        context.append(rightnbor)

                for child in context:

                    # Detect direct speech

                    # Reference to direct speech in previous sentence
                    if (tok == currentRoot and target != tok.head):
                        if target._.has_governing_subject:
                            thisDom = target
                        else:
                            dom = target.head
                            while not dom._.has_governing_subject \
                               and dom != dom.head:
                                dom = dom.head
                            thisDom = dom
                        speaker_refs = []
                        addressee_refs = []
                        if thisDom._.has_governing_subject:
                            for item in ResolveReference(hdoc[thisDom._.governing_subject], hdoc):
                                speaker_refs.append(item)
                            if thisDom in hdoc[thisDom._.governing_subject
                                               ].subtree:
                                thisDom = hdoc[thisDom._.governing_subject]
                            thisDom._.vwp_speaker_refs_ = speaker_refs

                            # Mark the addressee for the local domain,
                            # which will be the object of the preposition
                            # 'to' or the direct object
                            addressee_refs = \
                                self.markAddresseeRefs(thisDom,
                                                       tok,
                                                       addressee_refs)
                            thisDom._.vwp_addressee_refs_ = addressee_refs

                        thisDom._.vwp_direct_speech_verb_ = True
                        break

                    # we need punctuation BETWEEN the head and
                    # the complement for this to be indirect speech
                    elif (child.dep_ == 'punct'
                          and ((child.i < tok.i
                                and tok.i < target.i
                                and tok.dep_ == 'nsubj'
                                and tok.pos_ == 'VERB')
                               or (target.i < child.i
                                   and child.i < tok.i)
                               or (target.i > child.i
                                   and child.i > tok.i))):

                        if isRoot(target):
                            speaker_refs = []
                            addressee_refs = []

                        # Mark the speaker for the local domain, which will
                        # be the subject or agent, if one is available, and
                        # interpret first person pronouns within the local
                        # domain as referring to the subject of the domain
                        # predicate
                        subj = getSubject(target)

                        # TO-DO: add support for controlled subjects
                        # ("This is wrong", Jenna wanted to say.")

                        # Demoted subject of passive sentence
                        if subj is None:
                            subj = getPrepObject(tok, ['by'])

                        # Inverted subject construction
                        # ('spare me, begged the poor mouse')
                        if subj == tok and getObject(target) is not None:
                            subj = getObject(target)

                        # Mark this predicate as involving direct speech
                        # (complement clause with dependent punctuation is
                        # our best indicator of direct speech. That means
                        # we'll run into trouble when students do not punctuate
                        # correctly, but there's nothing to be done about it
                        # short of training a classifier on erroneous student
                        # writing.)

                        if subj is not None \
                           and (subj._.animate
                                or subj._.vwp_sourcetext
                                or subj.lower_ in ['they', 'it']
                                or subj.pos_ == 'PROPN'):
                            subj._.vwp_in_direct_speech_ = True
                            for child in subj.subtree:
                                child._.vwp_in_direct_speech_ = True
                            target._.vwp_direct_speech_verb_ = True
                        else:
                            continue

                        # If the quote doesn't encompass the complement,
                        # this isn't direct speech
                        if quotationMark(child) and not tok._.vwp_quoted:
                            continue

                        # Record the speaker as being the subject
                        if subj is not None:
                            target._.vwp_speaker_ = [subj.i]
                            tok._.vwp_speaker_ = [subj.i]
                            if subj.i not in speaker_refs:
                                speaker_refs.append(subj.i)
                            subjAnt = [hdoc[loc] for loc
                                       in ResolveReference(subj, hdoc)]
                            if subjAnt is not None:
                                for ref in subjAnt:
                                    if ref.i not in target._.vwp_speaker_:
                                        target._.vwp_speaker_.append(ref.i)
                                    if ref.i not in tok._.vwp_speaker_:
                                        tok._.vwp_speaker_.append(ref.i)
                                    if ref.i not in speaker_refs:
                                        speaker_refs.append(ref.i)

                        elif isRoot(tok.head):
                            target._.vwp_speaker_ = []
                            tok._.vwp_speaker_ = []

                        # Mark the addressee for the local domain,
                        # which will be the object of the preposition
                        # 'to' or the direct object
                        addressee_refs = \
                            self.markAddresseeRefs(target, tok, addressee_refs)

                        # If it does not conflict with the syntactic
                        # assignment, direct speech status is inherited
                        # from the node we detect as involving direct
                        # speech.
                        for descendant in target.subtree:
                            if descendant._.vwp_quoted:
                                # TO-DO: add block to prevent inheritance
                                # for embedded direct speech
                                if descendant.lower_ in \
                                   first_person_pronouns \
                                   and len(list(descendant.children)) == 0:
                                    descendant._.vwp_speaker_ = \
                                        target._.vwp_speaker_
                                    if descendant.i not in speaker_refs \
                                       and descendant.i not in addressee_refs:
                                        speaker_refs.append(descendant.i)
                        target._.vwp_speaker_refs_ = speaker_refs

                        # direct speech should be treated as quoted even if
                        # it isn't in quotation marks, as in 'This is good,
                        # I thought'. We only treat it as a quote if the
                        # tag is immediately adjacent to the punctuation mark.
                        if tok.right_edge.i + 2 == getSubject(tok.head).i \
                           or tok.right_edge.i + 2 == target.i:
                            for descendant in tok.subtree:
                                descendant._.vwp_quoted = True

                        for descendant in tok.subtree:
                            if descendant._.vwp_quoted:
                                if (descendant.lower_
                                    in second_person_pronouns
                                    and len(list(descendant.children)) == 0
                                    and descendant.i not in speaker_refs) \
                                   or (descendant.dep_ == 'vocative'
                                       and descendant.pos_ == 'NOUN'):
                                    descendant._.vwp_addressee_ = \
                                        tok.head._.vwp_addressee_
                                    if descendant.i not in addressee_refs:
                                        addressee_refs.append(descendant.i)
                        target._.vwp_addressee_refs_ = addressee_refs
                        break

            if (currentRoot is not None
                and lastRoot is not None
                and lastRoot._.vwp_quoted
                and (currentRoot.pos_ == 'VERB'
                     and getSubject(currentRoot) is None
                     and generalViewpointPredicate(currentRoot))):
                speaker_refs = []
                addressee_refs = []
                currentRoot._.vwp_direct_speech_verb_ = True
                for child in tok.children:
                    if child.dep_ == 'dobj':
                        subj = child
                        currentRoot._.vwp_direct_speech_verb_ = True
                        currentRoot._.vwp_addressee_ = []
                        currentRoot._.vwp_speaker_ = [subj.i]
                        child._.vwp_speaker_ = [subj.i]
                        if subj.i not in speaker_refs:
                            speaker_refs.append(subj.i)
                        subjAnt = [loc for loc
                                   in ResolveReference(subj, hdoc)]
                        if subjAnt is not None:
                            for ref in subjAnt:
                                if ref not in currentRoot._.vwp_speaker_:
                                    currentRoot._.vwp_speaker_.append(ref)
                                if ref not in child._.vwp_speaker_:
                                    child._.vwp_speaker_.append(ref)
                                if ref not in speaker_refs:
                                    speaker_refs.append(ref)
                        for descendant in lastRoot.subtree:
                            if descendant.lower_ in \
                               first_person_pronouns:
                                descendant._.vwp_speaker_ = \
                                    lastRoot._.vwp_speaker_
                                if descendant.i not in speaker_refs:
                                    speaker_refs.append(descendant.i)
                            if descendant.lower_ in \
                               second_person_pronouns \
                               or (descendant.dep_ == 'vocative'
                                   and descendant.pos_ == 'NOUN'):
                                if descendant._.vwp_addressee_ is None:
                                    descendant._.vwp_addressee_ = []
                                descendant._.vwp_addressee_.append(
                                    descendant.i)
                                if descendant.i not in speaker_refs:
                                    addressee_refs.append(descendant.i)
                        for addressee in addressee_refs:
                            if lastRoot._.vwp_addressee_refs_ is not None \
                                and addressee not in \
                                    lastRoot._.vwp_addressee_refs_:
                                if addressee not in \
                                   lastRoot._.vwp_addressee_refs_:
                                    lastRoot._.vwp_addressee_refs_.append(
                                       addressee)
                            else:
                                if addressee not in \
                                   lastRoot._.vwp_addressee_refs_:
                                    lastRoot._.vwp_addressee_refs_ = \
                                        [addressee]
                        for speaker in speaker_refs:
                            if lastRoot._.vwp_speaker_refs_ is not None \
                                and speaker not in \
                                    lastRoot._.vwp_speaker_refs_:
                                if speaker not in \
                                   lastRoot._.vwp_speaker_refs_:
                                    lastRoot._.vwp_speaker_refs_.append(
                                        speaker)
                            else:
                                if speaker not in \
                                   lastRoot._.vwp_speaker_refs_:
                                    lastRoot._.vwp_speaker_refs_ = [speaker]
                        currentRoot._.vwp_speaker_ = speaker_refs
                        currentRoot._.vwp_addressee_ = addressee_refs
                        currentRoot._.vwp_speaker_refs_ = speaker_refs
                        currentRoot._.vwp_addressee_refs_ = addressee_refs
                        break

            # A quotation following direct speech without identifier
            # can be assumed to be a continuation of the previous
            # direct speech. OR following an immediate introduction
            # by a communication/cognition/argument word

            if currentRoot is not None \
               and lastRoot is not None \
               and currentRoot._.vwp_quoted \
               and lastRoot._.vwp_direct_speech_verb_:

                currentRoot._.vwp_direct_speech_verb_ = True
                if lastRoot._.vwp_speaker_ is not None \
                   and len(lastRoot._.vwp_speaker_) > 0:
                    currentRoot._.vwp_speaker_ = lastRoot._.vwp_speaker_
                if lastRoot._.vwp_addressee_ is not None \
                   and len(lastRoot._.vwp_addressee_) > 0:
                    currentRoot._.vwp_addressee_ = lastRoot._.vwp_addressee_
                if lastRoot._.vwp_speaker_refs_ is not None \
                   and len(lastRoot._.vwp_speaker_refs_) > 0:
                    for item in lastRoot._.vwp_speaker_refs_:
                        if item not in speaker_refs:
                            speaker_refs.append(item)
                if lastRoot._.vwp_addressee_refs_ is not None \
                   and len(lastRoot._.vwp_addressee_refs_) > 0:
                    for item in lastRoot._.vwp_addressee_refs_:
                        if item not in addressee_refs:
                            addressee_refs.append(item)
                for descendant in tok.subtree:
                    # Direct speech status is inherited
                    # TO-DO: add block to prevent inheritance
                    # for embedded direct speech
                    if descendant.lower_ in first_person_pronouns:
                        descendant._.vwp_speaker_ = speaker_refs
                        if descendant.i not in speaker_refs:
                            speaker_refs.append(descendant.i)
                    if descendant.lower_ in second_person_pronouns \
                       or (descendant.dep_ == 'vocative'
                           and descendant.pos_ == 'NOUN'):
                        descendant._.vwp_addressee_ = lastRoot._.vwp_addressee_
                        if descendant.i not in addressee_refs:
                            addressee_refs.append(descendant.i)
                currentRoot._.vwp_speaker_refs_ = speaker_refs
                currentRoot._.vwp_addressee_refs_ = addressee_refs
                tok.head._.vwp_addressee_refs_ = addressee_refs
                tok.head._.vwp_speaker_refs_ = speaker_refs

            # Quoted text that contains first or second person
            # pronouns can be presumed to be direct speech
            if (isRoot(tok) and tok._.vwp_quoted) \
               or (tok._.vwp_quoted and '\n' in tok.head.text):
                if tok._.vwp_speaker_ is None:
                    tok._.vwp_speaker_ = []
                if tok._.vwp_addressee_ is None:
                    tok._.vwp_addressee_ = []

                if len(speaker_refs) > 0:
                    tok._.vwp_direct_speech_verb_ = True

                subtree = tok.subtree
                if isRoot(tok):
                    subtree = tok.head.subtree

                for descendant in subtree:
                    if descendant.lower_ in first_person_pronouns:
                        if descendant.i not in speaker_refs:
                            speaker_refs.append(descendant.i)
                        if descendant.i not in tok._.vwp_speaker_:
                            tok._.vwp_speaker_.append(descendant.i)
                        tok._.vwp_direct_speech_verb_ = True
                        tok._.vwp_speaker_ = [descendant.i]
                        tok._.vwp_speaker_refs_ = speaker_refs
                    if descendant.lower_ in second_person_pronouns \
                       or (descendant.dep_ == 'vocative'
                           and descendant.pos_ == 'NOUN'):
                        if descendant.i not in addressee_refs:
                            addressee_refs.append(descendant.i)
                        if descendant.i not in tok._.vwp_addressee_:
                            if descendant.i not in tok._.vwp_addressee_:
                                tok._.vwp_addressee_.append(descendant.i)
                        tok._.vwp_direct_speech_verb_ = True
                        tok._.vwp_addressee_ = [descendant.i]
                        tok._.vwp_addressee_refs_ = addressee_refs

                currentRoot._.vwp_speaker_refs_ = speaker_refs
                currentRoot._.vwp_addressee_refs_ = addressee_refs

                # TO-DO: Text with no specified viewpoint following
                # direct speech (including internal mental state predicates
                # like feel or believe used as direct speech tags) may be a
                # continuation of the direct speech segment. But we can't
                # resolve such cases without using a deeper NLP module that
                # is aware of other indicators of text macrostructure to
                # resolve implicit stance to the speaker or to some implied
                # speaker in a smaller text scope.

        self.set_direct_speech_spans(hdoc)
        return hdoc._.direct_speech_spans

    def perspectiveMarker(self, hdoc):
        """
         Find the viewpoint nominal that governs each viewpoint predicate
          in the document. For instance, in 'John is unlikely to win', there
          is no explicit viewpoint nominal, so the viewpoint is implicitly
          the speaker. But in 'According to Bill, John is unlikely to win',
          'Bill' governs 'unlikely', so the judgment of unlikelihood is
          attributed to Bill and not to the speaker.
        """

        subj = None

        for token in hdoc:

            # Evaluated role words when root of a domain take the
            # perspective of that domain (e.g., John deserves a raise)
            if token._.vwp_evaluated_role \
               and isRoot(token):
                token._.vwp_perspective_ = token._.head_perspective
                continue

            # Set viewpoint for emotional impact predicates
            # with an animate logical object
            if token._.vwp_emotional_impact \
               and getLogicalObject(token) is not None:
                obj = getLogicalObject(token)
                if obj._.animate:
                    self.registerViewpoint(hdoc, token, obj)
                    obj._.vwp_perspective_ = token._.vwp_perspective_
                if obj is not None \
                   and 'poss' in [child.dep_
                                  for child
                                  in obj.children
                                  if child._.animate]:
                    for child in obj.children:
                        if child.dep_ == 'poss' \
                           and child._.animate:
                            self.registerViewpoint(hdoc, token, child)
                            child._.vwp_perspective_ = token._.vwp_perspective_
                            token._.governing_subject_ = child.i
                            continue
                continue

            # Special case (evaluation predicates may take dative arguments)
            # Treat the object of the preposition as the viewpoint controller
            if token._.vwp_evaluation \
               or token._.vwp_emotion \
               or token._.vwp_emotional_impact:
                subj = getPrepObject(token, dative_preps)
                if subj is not None \
                   and (subj._.animate or subj._.vwp_sourcetext):
                    self.registerViewpoint(hdoc, token, subj)

                    # If this is adjectival, we need to mark the matrix
                    # verb/subject as being in the head's perspective
                    # because the to-phrase has purely local scope
                    if token.dep_ in object_predicate_dependencies \
                       or token.dep_ == 'acomp':
                        token.head._.vwp_perspective_ = \
                            self.getHeadDomain(token.head
                                               )._.head_perspective
                        if getSubject(token) is not None:
                            getSubject(token)._.vwp_perspective_ = \
                                self.getHeadDomain(token.head
                                                   )._.head_perspective

                    continue
                elif (self.getHeadDomain(token)._.vwp_perspective_ is not None
                      and len(self.getHeadDomain(token
                                                 )._.vwp_perspective_) > 0):
                    token._.vwp_perspective_ = \
                        self.getHeadDomain(token)._.vwp_perspective_
                    continue

            # Special case: Sentence-modifying prepositional phrases
            # to 'for me', 'to me', 'according to me' establish viewpoint
            # for their objects for that clause
            # Prefer sentence-initial if they appear there. Only use
            # late in the sentence if no viewpoint has yet been established
            if (token.i < token.head.i
                or token.head._.vwp_perspective_ is None
                or (token.head._.vwp_perspective_ is not None
                    and len(token.head._.vwp_perspective_) == 0)) \
               and (isRoot(token)
                    or token.dep_ == 'ccomp'
                    or token.dep_ == 'csubj'
                    or token.dep_ == 'csubjpass'
                    or token.dep_ == 'dep'
                    or token.dep_ == 'prep'
                    or token.dep_ == 'nsubj'):
                subj = getPrepObject(token, dative_preps)
                if token.dep_ != 'dep' \
                   and subj is not None \
                   and not subj._.location \
                   and (subj._.animate or subj._.vwp_sourcetext):
                    self.registerViewpoint(hdoc, token, subj)
                    self.registerViewpoint(hdoc, token.head, subj)
                    self.getHeadDomain(token)._.head_perspective = [subj.i]

            # Special case (npadvmod expressions like 'way'
            # in expressions like 'The way I see it, ...')
            # Treat the viewpoint of the npadvmod as viewpoint
            # for the sentence.
            if token.dep_ == 'npadvmod' and token.lemma_ in ['way', 'manner'] \
               and 'relcl' in [child.dep_ for child in token.children]:
                for child in token.children:
                    if child.dep_ == 'relcl':
                        subj = self.findViewpoint(
                            child, child, '', token, hdoc)
                        if subj is not None \
                           and (subj._.animate or subj._.vwp_sourcetext):
                            self.registerViewpoint(hdoc, token, subj)
                            self.getHeadDomain(token)._.head_perspective = \
                                [subj.i]
                            self.getHeadDomain(token)._.head_perspective = \
                                [subj.i]
                            break

            # Nouns used as predicates take the matrix viewpoint
            # for their domain
            if token.dep_ in object_predicate_dependencies \
               and token._.has_governing_subject:
                token._.vwp_perspective = token._.head_perspective
                continue

            # Evaluation words as subjects take the viewpoint of the
            # explicit or implicit speaker/thinker/sayer
            if token._.vwp_evaluation \
               and token.dep_ in ['nsubj', 'nsubpass']:
                # root clause, speaker perspective
                if token.head == token.head.head \
                   or (token.head.dep_ == 'conj'
                       and token.head.head == token.head.head.head):
                    token._.vwp_perspective = token._.head_perspective
                    continue
                else:
                    subj = self.findViewpoint(token,
                                              token.head,
                                              '',
                                              token,
                                              hdoc)
                    if subj is not None \
                       and (subj._.animate or subj._.vwp_sourcetext):
                        self.registerViewpoint(hdoc, token, subj)
                    continue

            # Subjective assessment words that are minor syntactic sisters
            # to the head of the domain take the viewpoint of the
            # explicit or implicit speaker/thinker/sayer
            if (token._.vwp_evaluation
                or token._.vwp_evaluated_role
                or token._.vwp_hedge
                or token._.vwp_possibility
                or token._.vwp_future
                or token._.subjectVerbInversion) \
               and token.dep_ in prehead_modifiers2:
                token._.vwp_perspective = \
                    self.getHeadDomain(token.head)._.head_perspective

                # root clause, speaker perspective
                if isRoot(token.head):
                    token._.vwp_perspective = \
                        token.head._.head_perspective

                if token.head.dep_ == 'conj' \
                   and isRoot(token.head.head):
                    token._.vwp_perspective = \
                        token.head.head._.head_perspective
                continue

            # Abstract nouns as (prepositional) objects of viewpoint predicates
            # e.g., I believe in censorship
            if token.pos_ == 'NOUN' \
               and token._.vwp_abstract \
               and token.dep_ == 'pobj' \
               and generalViewpointPredicate(token.head.head) \
               and token.head.head._.governing_subject is not None \
               and hdoc[token.head.head._.governing_subject]._.animate:
                self.registerViewpoint(hdoc,
                                       token.head.head,
                                       hdoc[
                                            token.head.head._.governing_subject
                                            ])
                self.registerViewpoint(hdoc,
                                       token,
                                       hdoc[
                                           token.head.head._.governing_subject
                                           ])

            # Abstract nouns as (direct) objects of viewpoint predicates
            # e.g., I detest censorship
            if token._.vwp_abstract \
               and token.dep_ == 'dobj' \
               and generalViewpointPredicate(token.head) \
               and token.head._.governing_subject is not None \
               and hdoc[token.head._.governing_subject]._.animate:
                self.registerViewpoint(hdoc,
                                       token.head,
                                       hdoc[token.head._.governing_subject])
                self.registerViewpoint(hdoc,
                                       token,
                                       hdoc[token.head._.governing_subject])
                continue

            # Viewpoint predicates
            if self.viewpointPredicate(token):
                # Set viewpoint for non-emotional impact predicates
                # with an animate governing subject
                if not token._.vwp_emotional_impact \
                   and token._.governing_subject is not None:
                    if token._.vwp_abstract \
                       and hdoc[token._.governing_subject]._.animate:
                        self.registerViewpoint(hdoc,
                                               token,
                                               hdoc[token._.governing_subject])
                    if token._.governing_subject is not None \
                       and 'poss' in [child.dep_
                                      for child
                                      in hdoc[token._.governing_subject
                                              ].children
                                      if child._.animate]:
                        for child in hdoc[token._.governing_subject].children:
                            if child.dep_ == 'poss' \
                               and child._.animate:
                                self.registerViewpoint(hdoc, token, child)
                                token._.governing_subject_ = child.i
                                continue
                    continue

            # Possessives for viewpoint predicates define viewpoint
            if token._.animate and token.dep_ == 'poss' \
               and (generalViewpointPredicate(token.head)
                    or token.head._.vwp_plan
                    or token.head._.vwp_emotion
                    or token.head._.vwp_emotional_impact):
                self.registerViewpoint(hdoc, token, token)
                self.registerViewpoint(hdoc, token.head, token)

            # General case -- find and register viewpoint expressions
            if self.viewpointPredicate(token):

                subj = self.findViewpoint(token,
                                          token,
                                          '',
                                          token,
                                          hdoc)
                if subj is not None \
                   and not token._.vwp_emotional_impact \
                   and (subj._.animate or subj._.vwp_sourcetext):
                    self.registerViewpoint(hdoc, token, subj)

        for token in hdoc:

            # Viewpoint predicates that are objects of certain
            # sentence-level PPs establish viewpoint for the
            # whole clause
            
            if token._.vwp_perspective_ is not None:
                if token.dep_ == 'pobj' \
                   and token.head.dep_ == 'prep' \
                   and (token.head.head.dep_ is None
                        or isRoot(token.head.head)
                        or token.head.head.dep_ in ['ccomp',
                                                    'csubj',
                                                    'csubjpass']):
                    if not isRoot(token.head.head):
                        token.head._.vwp_perspective_ = \
                            token.head.head.head._.vwp_perspective_
                    token.head.head._.vwp_perspective_ = token._.vwp_perspective_

            # Viewpoint predicates that are adverbial sentence-level modifiers
            # establish viewpoint for the whole clause
            while (token.head.pos_ != 'NOUN'
                   and isRoot(token)
                   and token._.vwp_perspective_ is not None
                   and ((token.dep_ == 'advcl'
                         or token.dep_ == 'advmod'
                         or token.dep_ == 'acomp'
                         or token.dep_ == 'npadvmod')
                        and not tough_complement(token)
                        and not raising_complement(token))):
                if token.head._.vwp_perspective_ is None \
                   or len(token.head._.vwp_perspective_) == 0:
                    token.head._.vwp_perspective_ = token._.vwp_perspective_
                    token = token.head
                else:
                    break
                continue

        # Spread the viewpoint assignment to all tokens within
        # the scope of the viewpoint markers we've found
        self.percolateViewpoint(getRoots(hdoc))

        for token in hdoc:
            if token._.vwp_perspective_ is None:
                token._.vwp_perspective_ = \
                    self.getHeadDomain(token)._.head_perspective
        return hdoc

    def markImplicitSubject(self, item, hdoc):
        """
         Use the dependency relations among words in the parse to
         identify the understood subject of predicates.
        """

        if item._.has_governing_subject_:
            return item._.governing_subject_

        found = False
        subj = getSubject(item)
        obj = getObject(item)

        ###################################################################
        # default governing subject is the explicit syntactic subject     #
        ###################################################################
        if subj is not None:
            item._.has_governing_subject_ = True
            item._.governing_subject_ = subj.i
            # We do not set found to True because this is potentially
            # overrideable

        ###################################################################
        # Assign subject of a relative clause to the modified noun        #
        # if subject is explicitly or implicitly a relative pronoun       #
        # but to the explicit subject otherwise                           #
        ###################################################################
        if item.dep_ == 'relcl' \
           and tensed_clause(item):
            # examples: the idea that upset him
            # the idea that he suggested /
            if subj and subj.tag_ in ['WDT', 'WP']:
                subj = item.head
                item._.has_governing_subject_ = True
                item._.governing_subject_ = subj.i
                found = True
            # the idea he gave me /
            # the idea that I like /
            # the idea I like
            elif subj:
                item._.has_governing_subject_ = True
                item._.governing_subject_ = subj.i
                found = True

        # infinitival adjectival clauses (my plan to win,
        # I have a plan to win, I developed a plan to win)
        # take their governing subject from the matrix object
        # in a double-object construction and from the matrix
        # subject in other object constructions
        if not found and item.dep_ == 'acl' \
           and not tensed_clause(item):
            if item.head._.has_governing_subject_:
                item._.has_governing_subject_ = True
                item._.governing_subject_ = \
                    item.head._.governing_subject_
                found = True
            elif item.dep_ in ['dobj', 'pobj']:
                if matrixobj is not None \
                   and item != matrixobj:
                    item._has_governing_subject_ = True
                    item._.governing_subject_ = matrixobj.i
                    found = True
                elif matrixsubj is not None:
                    item._.has_governing_subject_ = True
                    item._.governing_subject_ = matrixsubj.i
                    found = True

        if item.dep_ == 'conj' and getSubject(item) is not None:
            found = True

        ###############################################################
        # Conjuncts inherit their subject                             #
        # from the first conjunct                                     #
        ###############################################################
        dependency = item.dep_
        matrixitem = item
        while (matrixitem != matrixitem.head
               and matrixitem.dep_ == 'conj'
               and matrixitem != matrixitem.head
               and (getSubject(matrixitem) is None
                    or getSubject(matrixitem).tag_ in ['WDT', 'WP'])):
            matrixitem = matrixitem.head

        # Setup.
        if matrixitem.head._.has_governing_subject_:
            if hdoc[matrixitem.head._.governing_subject_
                    ]._.has_governing_subject_ \
               and not hdoc[matrixitem.head._.governing_subject_
                            ]._.animate \
               and (hdoc[matrixitem.head._.governing_subject_
                         ]._.concreteness is None
                    or hdoc[matrixitem.head._.governing_subject_
                            ]._.concreteness < 4):
                matrixsubj = \
                    hdoc[hdoc[matrixitem.head._.governing_subject_
                              ]._.governing_subject_]
            else:
                matrixsubj = hdoc[matrixitem.head._.governing_subject_]
        else:
            matrixsubj = getSubject(matrixitem.head)

            # Correction to get the head noun
            # not the subj. relative pronoun for
            # a relative clause
            if matrixsubj is not None \
               and matrixsubj.tag_ in ['WDT', 'WP']:
                matrixsubj = matrixsubj.head.head

        matrixobj = getObject(matrixitem.head)

        # Now inherit the governing subject from the first conjunct
        if item.dep_ == 'conj' \
           and matrixitem._.has_governing_subject_:
            item._.has_governing_subject_ = True
            item._.governing_subject_ = \
                matrixitem._.governing_subject_
            found = True

        ###################################################################
        # Embedded non-finite, subjectless predicates take the            #
        # nearest c-commanding argument as their subject, i.e.,           #
        # matrix object if present, matrix subject otherwise              #
        ###################################################################
        if not tensed_clause(item) \
           and matrixitem.dep_ in ['attr',
                                   'oprd',
                                   'prep',
                                   'dobj',
                                   'acomp',
                                   'ccomp',
                                   'pcomp',
                                   'xcomp',
                                   'dep']:

            if item._.has_governing_subject_:
                found = True

            # Matrix object is the governing subject
            elif (matrixsubj is not None
                  and matrixobj is not None
                  and matrixobj.tag_ not in ['WDT', 'WP']
                  and item != matrixobj):
                # Exception: indirect objects (first in two
                # dobj children) are not subjects of the
                # true direct object
                if item.dep_ != 'dobj':
                    if not containsDistinctReference(item, matrixobj, hdoc) \
                       or item.dep_ in ['ccomp', 'xcomp']:
                        item._.has_governing_subject_ = True
                        item._.governing_subject_ = matrixobj.i
                    else:
                        item._.has_governing_subject_ = False
                    found = True

            # Otherwise, matrix subject is the governing subject
            # with some restrictions on adjectives and nouns
            # (either they license raising/tough-movement constructions
            #  or, for nouns, we allow implicit subjects only in special
            # cases involving light verbs and abstract traits)
            elif (matrixsubj is not None
                  and (matrixobj is None or item != matrixobj)
                  and (item.pos_ in verbal_pos
                       or (item.pos_ in nominal_pos
                           and (item.head._.vwp_tough
                                or item.head._.vwp_raising
                                or item.head._.vwp_seem
                                or item.head._.vwp_cognitive
                                or item.head._.vwp_emotion
                                or item.head._.vwp_character
                                or item.dep_ in object_predicate_dependencies
                                or (item.dep_ == 'dobj'
                                    and (item._.abstract_trait
                                         or item.head.lemma_
                                         in getLightVerbs())))))):

                item._.has_governing_subject_ = True
                if item.pos_ == 'ADP':
                    item._.governing_subject_ = matrixitem.head.i
                else:
                    if not containsDistinctReference(item, matrixsubj, hdoc) \
                       or item.dep_ in ['ccomp', 'xcomp']:
                        item._.has_governing_subject_ = True
                        item._.governing_subject_ = matrixsubj.i
                    else:
                        item._.has_governing_subject_ = False
                    found = True

            if matrixsubj is not None \
               and matrixobj is not None \
               and (matrixobj.head.lemma_ in getLightVerbs()
                    or matrixobj.head._.vwp_cognitive
                    or matrixobj.head._.vwp_emotion):
                matrixobj._.has_governing_subject_ = True
                matrixobj._.governing_subject_ = matrixsubj.i

        # The governing subject of an attributive adjective
        # or a preposition or a clause modifying a noun
        # is the noun is modifies
        if not found \
           and item.dep_ in adjectival_mod_dependencies \
           and item.head.pos_ in nominal_pos:
            item._.has_governing_subject_ = True
            item._.governing_subject_ = item.head.i
            found = True

        # The governing subject of a preposition or adverb
        # modifying a verb is that verb's governing subject,
        # except for sentence adverbs (which are listed by type)
        # and transition words
        if not found \
           and (item.dep_ in verbal_mod_dependencies
                or item.pos_ == 'ADV'):
            if item.dep_ in ['neg'] \
               or item._.vwp_hedge \
               or item._.transition:
                item._.has_governing_subject_ = True
                item._.governing_subject_ = item.head.i
                found = True
            elif (item.head.pos_ in verbal_pos
                  and item.head._.has_governing_subject_
                  and not item._.vwp_necessity
                  and not item._.vwp_certainty
                  and not item._.vwp_probability
                  and not item._.vwp_importance
                  and not item._.vwp_risk
                  and not item._.vwp_expectancy
                  and not item._.vwp_emphasis
                  and not item._.vwp_information
                  and not item._.transition):
                item._.has_governing_subject_ = True
                item._.governing_subject_ = item.head._.governing_subject_
                found = True
            # otherwise it is the modified word
            else:
                item._.has_governing_subject_ = True
                item._.governing_subject_ = item.head.i
                found = True

        # Prepositional and object complements that are gerunds w/o
        # a marked subject take the subject of the closest ancestor
        # with a subject or object commanding them
        if not found \
           and item.dep_ in ['ccomp', 'pcomp'] \
           and item.tag_ == 'VBG' \
           and getSubject(item) is None:
            last = item
            head = item.head
            while (head != head.head
                   and getObject(head) is None
                   and getSubject(head) is None):
                last = head
                head = head.head
            obj = getObject(head)
            subj = getSubject(head)
            if obj is not None and obj != last:
                item._.has_governing_subject_ = True
                item._.governing_subject_ = obj.i
                found = True
            elif subj is not None:
                item._.has_governing_subject_ = True
                item._.governing_subject_ = subj.i
                found = True

        # Participles also take the subject of the clause they modify
        if not found \
           and item.dep_ in ['advcl'] \
           and item.tag_ == 'VBG' \
           and (getSubject(item) is None
                or item.pos_ != 'VERB'):
            last = item
            head = item.head
            while (head != head.head
                   and getSubject(head) is None):
                last = head
                head = head.head
            obj = getObject(head)
            subj = getSubject(head)
            if obj is not None and obj != last and obj._.has_governing_subject_:
                item._.has_governing_subject_ = True
                item._.governing_subject_ = obj._.governing_subject_
                found = True
            elif subj is not None and subj._.has_governing_subject_:
                item._.has_governing_subject_ = True
                item._.governing_subject_ = subj._.governing_subject_
                found = True

        for child in item.children:
            self.markImplicitSubject(child, hdoc)

        return item._.governing_subject_

    def isHeadDomain(self, node):
        """
        Definition of what nodes count as head domains for viewpoint
        """

        # e.g., 'make us lazier'
        if node.dep_ in ['ccomp', 'oprd'] \
           and takesBareInfinitive(node.head):
            return True

        # e.g., 'than ever before'
        if node.dep_ in ['pcomp'] and node.pos_ == 'ADV':
            return True

        # e.g., 'with out the invention of the internet'
        if node.dep_ == 'pcomp' \
           and node.tag_ == 'RP':
            return True

        # conjuncts are head domain if the word they're conjoined with is
        if node.dep_ == 'conj' \
           and self.isHeadDomain(node.head):
            return True

        if not isRoot(node) \
           and node.dep_ not in clausal_complements \
           and not (tensed_clause(node)
                    and node.dep_ in ['advcl', 'relcl']) \
           and not (tensed_clause(node)
                    and node.dep_ == 'conj'
                    and not self.isHeadDomain(node.head)) \
           and not (node.dep_ == 'acl'
                    and node.tag_ in ['VBD', 'VBN']) \
           and not (node.dep_ == 'acl'
                    and node.tag_ in ['VBG']
                    and node.head.pos_ == 'NOUN') \
           and not (node.dep_ in ['amod']
                    and (node._.vwp_evaluation
                         or generalViewpointPredicate(node))) \
            or (node.pos_ == 'VERB'
                and not tensed_clause(node)
                and node.dep_ in clausal_complements):
            return True
        return False

    def getHeadDomain(self, node):
        """
        Find the head of a viewpoint domain.
        """
        start = node.text

        # Complements and relative clauses are opaque to viewpoint
        while self.isHeadDomain(node) and node != node.head:
            node = node.head
        return node

    def percolateViewpoint(self, nodes: list, barriers=[]):
        """
         Viewpoint is defined relative to head domains (basically,
         tensed clauses). We need to associate each node with the
         viewpoint of its immediately-dominating head domain, which
         this function does by assigning each token to the perspective
         associated with its domain head. Normally, this will be the
         subject of each tensed clause, if the verb is a viewpoint
         predicate. E.g., in 'Jenna believes that Mary is happy that
         Bill is going to school', the proposition 'Bill is going to
         school' is assessed from Mary's perspective, whereas the
         proposition 'Mary is happy that Bill is going to school'
          is assessed from Jenna's perspective.
        """

        for node in nodes:

            if node.head == self.getHeadDomain(node.head) \
               and node.head.dep_ == 'conj' \
               and node.head.i not in barriers:
                barriers.append(node.head.i)
            elif (node == self.getHeadDomain(node)
                  and node.dep_ == 'conj'
                  and node.head.i not in barriers):
                barriers.append(node.i)
            elif (node.dep_ in ['csubj', 'csubjpass']
                  and node.head.i not in barriers):
                barriers.append(node.i)

            if node._.vwp_perspective_ is None:
                if not isRoot(self.getHeadDomain(node)) \
                   and (node != self.getHeadDomain(node)
                        or node.dep_ not in prehead_modifiers2):

                    if self.getHeadDomain(node).i not in barriers:
                        node._.vwp_perspective_ = \
                            self.getHeadDomain(node).head._.vwp_perspective_
                elif node == self.getHeadDomain(node):
                    node._.vwp_perspective_ = \
                        self.getHeadDomain(node.head)._.head_perspective
                else:
                    node._.vwp_perspective_ = \
                        self.getHeadDomain(node)._.head_perspective
            else:
                if (node._.governing_subject_ is None
                   or len(node._.vwp_perspective_)) == 0 \
                   and node.head.lemma_ in getLightVerbs():
                    node.head._.vwp_perspective_ = node._.vwp_perspective_
            for child in node.children:

                if child._.vwp_perspective_ is not None:
                    self.percolateViewpoint([child], barriers)
                    continue

                if child.dep_ not in prehead_modifiers2:

                    found = False
                    if self.getHeadDomain(child).i not in barriers:
                        if node._.vwp_perspective_ is not None:
                            child._.vwp_perspective_ = node._.vwp_perspective_
                        elif (not found and node != self.getHeadDomain(node)
                              and node.i < self.getHeadDomain(node).i):
                            if child._.vwp_perspective_ is not None \
                               and not isRoot(self.getHeadDomain(node)) \
                               and len(child._.vwp_perspective_) == 0:

                                child._.vwp_perspective_ = \
                                         self.getHeadDomain(
                                             node).head._.vwp_perspective_
                            elif child._.vwp_perspective_ is None:
                                child._.vwp_perspective_ = \
                                    self.getHeadDomain(child
                                                       )._.head_perspective
                        elif (child.i < node.i
                              and child.i < self.getHeadDomain(node).i):
                            if child._.vwp_perspective_ is None:
                                child._.vwp_perspective_ = \
                                    self.getHeadDomain(child
                                                       )._.head_perspective
                else:
                    child._.vwp_perspective_ = \
                        self.getHeadDomain(child)._.head_perspective

                self.percolateViewpoint([child], barriers)

    def set_direct_speech_spans(self, hdoc):
        """
          Find direct speech spans (recognized because the head word
          of the span has the extension attribute vwp_direct_speech
          set true) and record the token index for the speaker,
          the start of the direct speech segment, and the end of
          the direct speech segment.
        """

        dspans = []
        for token in hdoc:
            if token._.vwp_direct_speech_verb_:
                speaker = token._.vwp_speaker_refs_
                addressee = token._.vwp_addressee_refs_
                left = token.left_edge
                right = token.right_edge

                # Edge case: quotation mark for a following quote
                # Adjust span boundaries to include the quotation
                # mark in the following not the preceding span
                if left.i > 0 \
                   and quotationMark(hdoc[left.i - 1]) \
                   and hdoc[left.i]._.vwp_quoted:
                    left = hdoc[left.i - 1]
                if right.i + 1 < len(hdoc) and right.i > 1:
                    if quotationMark(hdoc[right.i]) \
                       and not hdoc[right.i - 1]._.vwp_quoted:
                        hdoc[right.i]._.vwp_quoted = True
                        right = hdoc[right.i - 1]

                # Define spans
                if speaker is not None \
                   and addressee is not None \
                   and len(speaker) > 0 \
                   and len(addressee) > 0:
                    dspans.append([speaker,
                                  addressee,
                                  left.i,
                                  right.i])
                elif speaker is not None and len(speaker) > 0:
                    dspans.append([speaker,
                                  [],
                                  left.i,
                                  right.i])
                elif addressee is not None and len(addressee) > 0:
                    dspans.append([[],
                                  addressee,
                                  left.i,
                                  right.i])
        lastSpeaker = None
        lastAddressee = None
        lastItem = None
        newSpans = []
        # Merge lists of references so the same list
        # is assigned to spans with references
        # included in a later list
        for span in reversed(dspans):
            newItem = [span[0], span[1], [[span[2], span[3]]]]
            if lastSpeaker is None:
                newItem[0] = list(sorted(span[0]))
                newItem[1] = list(sorted(span[1]))
                if newItem not in newSpans:
                    newSpans.append(newItem)
            else:
                included = True
                for item in span[0]:
                    if item not in lastSpeaker:
                        included = False
                if included:
                    if span[2] <= newSpans[len(newSpans) - 1][2][0][0] \
                       and span[3] >= newSpans[len(newSpans) - 1][2][0][1]:
                        newSpans[len(newSpans) - 1][2][0][0] = span[2]
                        newSpans[len(newSpans) - 1][2][0][1] = span[3]
                    else:
                        newSpans[len(newSpans) - 1][2].insert(0,
                                                              [span[2],
                                                               span[3]])
                else:
                    newItem[0] = list(sorted(span[0]))
                    newItem[1] = list(sorted(span[1]))
                    if newItem not in newSpans:
                        newSpans.append(newItem)
            lastSpeaker = span[0]
            lastAddressee = span[1]
            lastItem = newItem
        
        newList = []
        for record in reversed(newSpans):
                newList.append(record)

        # add the token level flag and do the standard format output       
        hdoc._.direct_speech_spans = []
        for span in newList:
            (speaker, addressee, subspans) = span
            for subspan in subspans:
                [left, right] = subspan
                entry = newSpanEntry('direct speech',
                                     left,
                                     right,
                                     hdoc,
                                     [speaker, addressee])
                hdoc._.direct_speech_spans.append(entry)
            locs = span[2]
            for loc in locs:
                leftEdge = loc[0]
                rightEdge = loc[1]
                for item in hdoc[leftEdge:rightEdge]:
                    item._.vwp_in_direct_speech_ = True
        return hdoc._.direct_speech_spans

    def vwp_egocentric(self, hdoc):
        """
         Viewpoint domains that contain evaluation language like should
         or perhaps with explicit or implicit first-person viewpoint
         count as egocentic. 'Unfortunately, Jenna came in last' ->
         egocentric since implicitly it is the speaker who views the
         event as unfortunate
        """
        count = 0
        domainList = []
        entityInfo = []
        for token in hdoc:
            if (token._.vwp_evaluation or token._.vwp_hedge) \
               and len(token._.vwp_perspective) == 0 \
               and self.getHeadDomain(token).i not in domainList:
                domainList.append(self.getHeadDomain(token).i)
            for perspective in token._.vwp_perspective:
                if ((token._.vwp_evaluation or token._.vwp_hedge)
                    and (hdoc[perspective].lower_
                         in first_person_pronouns)
                    and (self.getHeadDomain(token).i
                         not in domainList)):
                    domainList.append(self.getHeadDomain(token).i)
        for token in hdoc:
            entry = newTokenEntry('egocentric', token)
            if self.getHeadDomain(token).i in domainList:
                entry['value'] = True
            else:
                entry['value'] = False
            entityInfo.append(entry)
        return entityInfo

    def vwp_allocentric(self, doc):
        count = 0
        domainList = []
        entityInfo = []
        for token in doc:
            entry = newTokenEntry('egocentric', token)
            include = True
            if len(token._.vwp_perspective) == 0:
                include = False
            else:
                for perspective in token._.vwp_perspective:
                    if perspective is None \
                       or (doc[perspective].lower_
                           in first_person_pronouns) \
                       or (doc[perspective].lower_
                           in second_person_pronouns):
                        include = False
            if include:
                entry['value'] = True
            else:
                entry['value'] = False
            entityInfo.append(entry)
        return entityInfo

    def set_perspective_spans(self, hdoc):
        """
         Create structures identifying which tokens have been
         assigned to which point of view

         'implicit' means there is no explicit viewpoint noun
         controlling this item. That means that it is probably,
         implicitly, the speaker.

         'explicit_1' means that there is an explicit first person
         pronoun controlling this item, in a context where the first
         person pronoun should be interpreted as the speaker

         'explicit_2' means that there is an explicit second person
         pronoun controlling this item, in a context where the second
         person pronoun should be interpreted as the addressee

         'explicit_3' means that there is an explicit third person
         viewpoint nominal controlling this item. We list individual
         token offsets under the offset of the controlling viewpoint
         nominal.

         We create the following data structures
         1. a basic perspective dictionary that shows whose viewpoint
            each token is interpreted from
         2. a stance marker dictionary that shows which words cue
            subjective evaluation of propositions from particular points
            of view.
         3. An emotion marker dictionary that shows which emotions are
            associated with which animate nominals in the text.
         4. a character marker dictionary that shows which character
            traits are associated with which animate nominals in the text.
         5. a propositional attitude dictionary that identifies words
            like think, feel, etc. that express attitudes toward propositions
            by agents.

        """
        pspans = {}
        pspans['implicit'] = {}
        pspans['implicit_1'] = []
        pspans['implicit_3'] = []
        pspans['explicit_1'] = []
        pspans['explicit_2'] = []
        pspans['explicit_3'] = {}

        stance_markers = {}
        stance_markers['implicit'] = {}
        stance_markers['implicit_1'] = []
        stance_markers['implicit_3'] = []
        stance_markers['explicit_1'] = []
        stance_markers['explicit_2'] = []
        stance_markers['explicit_3'] = {}

        for token in hdoc:

            referentID = ResolveReference(token, hdoc)
            if token.i not in referentID:
                referentID.append(token.i)

            #self.mark_argument_words(token, hdoc)
            hdeps = [child.dep_ for child in token.head.children]
            stance_markers, pspans = \
                self.stance_perspective(token,
                                        hdoc,
                                        referentID,
                                        stance_markers,
                                        pspans,
                                        hdeps)

        hdoc._.vwp_perspective_spans_ = \
            self.cleanup_propositional_attitudes(
                pspans, hdoc, "perspective_spans")
        hdoc._.vwp_stance_markers_ = \
            self.cleanup_propositional_attitudes(
                stance_markers, hdoc, "stance_markers")

    def mark_argument_words(self, token, hdoc):

        tp = hdoc._.transition_word_profile
        for item in tp[3]:
            if item[4] not in ['temporal', 'PARAGRAPH']:
                if item[2] == item[3]:
                    hdoc[item[2]]._.vwp_argumentation_ = True
                    hdoc[item[2]]._.transition = True
                else:
                    for i in range(item[2], item[3] + 1):
                        hdoc[i]._.vwp_argumentation_ = True
                        hdoc[i]._.transition = True
        if (token._.vwp_cognitive
           or token._.vwp_communication
           or token._.vwp_argument
           or token._.vwp_evaluation) \
           and token.dep_ == 'pobj' \
           and token.head.head._.abstract_trait:
            if token.head.head.dep_ == 'pobj':
                if (token.head.head.head.head._.vwp_cognitive
                    or token.head.head.head.head._.vwp_communication
                    or token.head.head.head.head._.vwp_argument
                    or token.head.head.head.head._.vwp_information
                    or token.head.head.head.head._.vwp_cognitive
                    or (token.head.head.head.dep_ != 'conj'
                        and (token.head.head.head.head._.vwp_abstract
                             or token.head.head.head.head._.vwp_possession
                             or token.head.head.head.head._.vwp_cause
                             or token.head.head.head.head._.vwp_relation))):
                    token._.vwp_argumentation_ = True
                    token.head._.vwp_argumentation_ = True
                    token.head.head._.vwp_argumentation_ = True
                    token.head.head.head._.vwp_argumentation_ = True
                    token.head.head.head.head._.vwp_argumentation_ = True
                    for child in token.head.head.children:
                        if child.pos_ in ['DET'] \
                           or child.tag_ in ['WP', 'WP$', 'JJR', 'JJS']:
                            child._.vwp_argumentation_ = True

            elif token.head.head.dep_ == 'dobj':
                if (generalViewpointPredicate(token.head.head.head)
                    or token.head.head.head._.vwp_information
                    or (token.head.head.dep_ != 'conj'
                        and (token.head.head.head._.vwp_abstract
                             or token.head.head.head._.vwp_possession
                             or token.head.head.head._.vwp_cause
                             or token.head.head.head._.vwp_relation))):
                    token._.vwp_argumentation_ = True
                    token.head._.vwp_argumentation_ = True
                    token.head.head._.vwp_argumentation_ = True
                    token.head.head.head._.vwp_argumentation_ = True
                    for child in token.head.head.children:
                        if child.pos_ in ['DET'] \
                           or child.tag_ in ['WP', 'WP$', 'JJR', 'JJS']:
                            child._.vwp_argumentation_ = True
        if token.dep_ == 'pobj' \
           and token.head is not None \
           and token.head.dep_ == 'prep' \
           and token.head.head is not None \
           and token.head.head.head is not None \
           and token.head.head.dep_ == 'advmod' \
           and generalViewpointPredicate(token) \
           and generalViewpointPredicate(token.head.head.head):
            if (generalViewpointPredicate(token)
                or token._.vwp_information
                or (token.dep_ != 'conj'
                    and (token._.vwp_abstract
                         or token._.vwp_possession
                         or token._.vwp_cause
                         or token._.vwp_relation))):
                token._.vwp_argumentation_ = True
            token.head._.vwp_argumentation_ = True
            token.head.head._.vwp_argumentation_ = True
            token.head.head.head._.vwp_argumentation_ = True

        if token.dep_ == 'pobj' \
           and token.head is not None \
           and token.head.dep_ == 'prep' \
           and token.head.head is not None \
           and token.head.head.pos_ in ['VERB',
                                        'NOUN',
                                        'ADJ',
                                        'ADV']:
            for child in token.head.head.children:
                if child._.vwp_evaluation \
                   or child._.vwp_hedge \
                   or generalViewpointPredicate(child):
                    if (token._.vwp_cognitive
                        or generalViewpointPredicate(token)
                        or token._.vwp_information
                        or (child.dep_ != 'conj'
                            and (token._.vwp_abstract
                                 or token._.vwp_possession
                                 or token._.vwp_cause
                                 or token._.vwp_relation))):
                        token._.vwp_argumentation_ = True
                        token.head._.vwp_argumentation_ = True
                        child._.vwp_argumentation_ = True
                    if token.dep_ != 'conj' \
                       and token.head.dep_ != 'conj' \
                       and (token._.vwp_cognitive
                            or generalViewpointPredicate(token.head.head)
                            or token.head.head._.vwp_information
                            or (token.head.dep_ != 'conj'
                                and (token.head.head._.vwp_abstract
                                     or token.head.head._.vwp_possession
                                     or token.head.head._.vwp_cause
                                     or token.head.head._.vwp_relation))):
                        token.head.head._.vwp_argumentation_ = True
                        child._.vwp_argumentation_ = True

        if token.dep_ in ['pobj', 'advmod'] \
           and isRoot(token.head.head) \
           and (generalViewpointPredicate(token)
                or token._.vwp_evaluation
                or token._.vwp_hedge):
            for child in token.head.head.children:
                if (generalViewpointPredicate(child)
                    or child._.vwp_evaluation
                    or child._.vwp_hedge
                    or child._.vwp_information
                    or (child.dep_ != 'conj'
                        and (child._.vwp_abstract
                             or child._.vwp_possession
                             or child._.vwp_cause
                             or child._.vwp_relation))):
                    token._.vwp_argumentation_ = True
                    child._.vwp_argumentation_ = True

        if token._.has_governing_subject_ \
           and (token._.vwp_evaluation
                or token._.vwp_hedge) \
           and (generalViewpointPredicate(hdoc[token._.governing_subject_])
                or hdoc[token._.governing_subject_]._.vwp_information
                or (token.dep_ != 'conj'
                    and (hdoc[token._.governing_subject_]._.vwp_abstract
                         or hdoc[token._.governing_subject_]._.vwp_possession
                         or hdoc[token._.governing_subject_]._.vwp_relation))):
            token._.vwp_argumentation_ = True
            hdoc[token._.governing_subject_]._.vwp_argumentation_ = True
            for child in hdoc[token._.governing_subject_].children:
                if child.lemma_ not in core_temporal_preps \
                    and (child.pos_ in ['DET', 'AUX']
                         or child.tag_ in function_word_tags
                         or stance_adverb(child)
                         or child.lemma_ in personal_or_indefinite_pronoun
                         or generalViewpointPredicate(child)
                         or child._.vwp_information
                         or child._.vwp_evaluation
                         or child._.vwp_hedge
                         or (child.dep_ != 'conj'
                             and (child._.vwp_abstract
                                  or child._.vwp_possession
                                  or child._.vwp_cause
                                  or child._.vwp_relation
                                  or child.dep_ in ['neg']))):
                    child._.vwp_argumentation_ = True

        if token.dep_ == 'amod' \
           and token._.vwp_evaluation \
           and token.head.dep_ in subject_or_object_nom \
           and (isRoot(token.head.head)
                or generalViewpointPredicate(token)):
            token._.vwp_argumentation_ = True

        if token.dep_ in adjectival_predicates \
           and (token._.vwp_evaluation
                or token._.vwp_raising
                or token._.vwp_hedge
                or generalViewpointPredicate(token)):
            for child in token.head.children:
                if child.dep_ in clausal_complements:
                    if generalViewpointPredicate(child) \
                       or child._.vwp_information \
                       or child._.vwp_evaluation \
                       or child._.vwp_hedge \
                       or (child.dep_ != 'conj'
                           and (child._.vwp_abstract
                                or child._.vwp_possession
                                or child._.vwp_possession
                                or child._.vwp_cause
                                or child._.vwp_relation
                                or child.dep_ in ['neg'])):
                        token._.vwp_argumentation_ = True
                        child._.vwp_argumentation_ = True
                    for grandchild in child.children:
                        if child._.vwp_information \
                           or grandchild._.vwp_communication \
                           or grandchild._.vwp_cognitive \
                           or (grandchild.dep_ != 'conj'
                                and (grandchild._.vwp_abstract
                                     or grandchild._.vwp_possession
                                     or grandchild._.vwp_evaluation
                                     or grandchild._.vwp_hedge
                                     or grandchild._.vwp_possession
                                     or grandchild.dep_ in ['neg'])):
                            token._.vwp_argumentation_ = True
                            grandchild._.vwp_argumentation_ = True

        if token.dep_ == 'amod' \
           and (token._.vwp_evaluation
                or token._.vwp_hedge) \
           and token.head.dep_ == 'pobj' \
           and token.head.head.dep_ == 'prep' \
           and token.head.head.head is not None \
           and token.head.head.head.pos_ in ['VERB', 'NOUN', 'ADJ', 'ADV']:
            for child in token.head.head.head.children:
                if child._.vwp_evaluation \
                   or child._.vwp_hedge \
                   or generalViewpointPredicate(child):
                    token._.vwp_argumentation_ = True
                    if token.head._.vwp_evaluation \
                       or token.head._.vwp_hedge \
                       or generalViewpointPredicate(token.head):
                        token.head._.vwp_argumentation_ = True
                        token.head.head._.vwp_argumentation_ = True
                    if token.head.head.head._.vwp_evaluation \
                       or token.head.head.head._.vwp_hedge \
                       or generalViewpointPredicate(token.head.head.head):
                        token.head.head.head._.vwp_argumentation_ = True
                    child._.vwp_argumentation_ = True

        if token.dep_ == 'amod' \
           and (token._.vwp_evaluation
                or token._.vwp_hedge) \
           and (in_modal_scope(token.head)
                or not token.head._.in_past_tense_scope) \
           and (generalViewpointPredicate(token.head)
                or token.head._.vwp_information
                or (token.dep_ != 'conj'
                    and (token.head._.vwp_abstract
                         or token.head._.vwp_possession
                         or token.head._.vwp_relation))) \
           and token.head._.is_academic:
            token._.vwp_argumentation_ = True
            token.head._.vwp_argumentation_ = True

        if token.lemma_ in quantifying_determiners \
           and (generalViewpointPredicate(token.head)
                or token.head._.vwp_information
                or (token.dep_ != 'conj'
                    and (token.head._.vwp_abstract
                         or token.head._.vwp_possession
                         or token.head._.vwp_cause
                         or token.head._.vwp_relation
                         or token.head._.abstract_trait))):
            for child in token.head.children:
                if token != child:
                    if stancePredicate(child):
                        token._.vwp_argumentation_ = True
                        token.head._.vwp_argumentation_ = True
                        child._.vwp_argumentation_ = True

        if ((token._.vwp_evaluation or token._.vwp_hedge)
            and token.head.dep_ not in ['conj']
            and (generalViewpointPredicate(token.head.head))):
            token._.vwp_argumentation_ = True
            token.head.head._.vwp_argumentation_ = True

        if stancePredicate(token):
            if token.dep_ in subject_or_object_nom:
                for child in token.head.children:
                    if child.dep_ in clausal_complements:
                        token._.vwp_argumentation_ = True
                        token.head._.vwp_argumentation_ = True
                        for grandchild in child.children:
                            if grandchild.dep_ == 'mark':
                                grandchild._.vwp_argumentation_ = True

            if token.dep_ in auxiliary_or_adverb \
               and (isRoot(token.head)
                    and stancePredicate(token.head)):
                token._.vwp_argumentation_ = True
                token.head._.vwp_argumentation_ = True
                for child in token.head.children:
                    if child.pos_ in ['AUX', 'ADV']:
                        child._.vwp_argumentation_ = True

            if token.dep_ in auxiliary_or_adverb \
               and (token._.vwp_evaluation or token._.vwp_hedge):
                for child in token.head.children:
                    if child.dep_ != 'conj':
                        for grandchild in child.children:
                            if (grandchild.dep_ != 'conj'
                                and stancePredicate(grandchild)):
                                token._.vwp_argumentation_ = True
                                grandchild._.vwp_argumentation_ = True
                            for ggrandchild in grandchild.children:
                                if (ggrandchild.dep_ != 'conj'
                                    and (stancePredicate(ggrandchild)
                                         or ggrandchild._.vwp_probability)):
                                    token._.vwp_argumentation_ = True
                                    ggrandchild._.vwp_argumentation_ = True

            if token.dep_ in auxiliary_or_adverb:
                for child in token.children:
                    if child.dep_ in adjectival_predicates \
                       and stancePredicate(child):
                        token._.vwp_argumentation_ = True
                        child._.vwp_argumentation_ = True

            if token.dep_ is None \
               or isRoot(token) \
               or (isRoot(token.head)
                   and token.dep_ == 'attr') \
               or stancePredicate(token.head):
                for child in token.children:
                    if clausal_subject_or_complement(child):
                        token._.vwp_argumentation_ = True
                        token.head._.vwp_argumentation_ = True
                        for child in token.head.children:
                            if child.lemma_ not in core_temporal_preps \
                               and (child.pos_ in ['DET', 'AUX']
                                    or child.tag_ in function_word_tags
                                    or stance_adverb(child)
                                    or child.lemma_
                                    in personal_or_indefinite_pronoun
                                    or generalViewpointPredicate(child)
                                    or child._.vwp_information
                                    or (child.dep_ != 'conj'
                                        and (child._.vwp_abstract
                                             or child._.vwp_possession
                                             or child._.vwp_relation
                                             or child._.vwp_cause
                                             or child.dep_ in ['neg']))):
                                child._.vwp_argumentation_ = True

                        if token.i + 1 < len(token.doc) \
                           and token.nbor(1) is not None \
                           and token.nbor(1).dep_ == 'mark':
                            token.nbor(1)._.vwp_argumentation_ = True

        if token.dep_ == 'amod' \
           and (token._.vwp_evaluation
                or token._.vwp_hedge) \
           and (in_modal_scope(token.head)
                or not token.head._.in_past_tense_scope) \
           and stancePredicate(token.head):
            token._.vwp_argumentation_ = True
            token.head._.vwp_argumentation_ = True

        if token.dep_ == 'amod' \
           and (token._.vwp_evaluation
                or token._.vwp_hedge) \
           and (in_modal_scope(token.head.head)
                or not token.head.head._.in_past_tense_scope) \
           and stancePredicate(token.head.head):
            token._.vwp_argumentation_ = True
            token.head.head._.vwp_argumentation_ = True

        if token.dep_ == 'amod' \
           and (token._.vwp_evaluation
                or token._.vwp_hedge):
            if (token.head.dep_ == 'pobj'
                and (in_modal_scope(token.head.head.head)
                     or not token.head.head.head._.in_past_tense_scope)
                and token.head.head.head is not None
                and stancePredicate(token.head.head.head)):
                token._.vwp_argumentation_ = True
                token.head.head.head._.vwp_argumentation_ = True

        if token.dep_ == 'prep' \
           and (in_modal_scope(token.head)
                or not token.head._.in_past_tense_scope) \
           and (generalViewpointPredicate(token.head)
                or token.head._.vwp_information
                or (token.dep_ != 'conj'
                    and (token.head._.vwp_abstract
                         or token.head._.vwp_possession
                         or token.head._.vwp_cause
                         or token.head._.vwp_relation))):
            for child in token.children:
                if (child.dep_ == 'pobj'
                    and stancePredicate(child)):
                    token._.vwp_argumentation_ = True
                    token.head._.vwp_argumentation_ = True
                    child._.vwp_argumentation_ = True

        if (token.dep_ in complements
            and (generalViewpointPredicate(token.head)
                 or token.head._.vwp_information
                 or (token.dep_ != 'conj'
                     and (token.head._.vwp_abstract
                          or token.head._.vwp_possession
                          or token.head._.vwp_cause
                          or token.head._.vwp_relation)))):
            for child in token.children:
                if child.dep_ == 'dobj' \
                   and stancePredicate(child):
                    token._.vwp_argumentation_ = True
                    token.head._.vwp_argumentation_ = True
                    child._.vwp_argumentation_ = True

        if generalViewpointPredicate(token):
            for offset in getLinkedNodes(token):
                if (token.lower_ != hdoc[offset].lower_
                    and (token.head.dep_ is None
                         or isRoot(token.head)
                         or isRoot(hdoc[offset].head))):
                    if hdoc[offset].dep_ == 'prep':
                        for child in hdoc[offset].children:
                            if generalViewpointPredicate(child):
                                token._.vwp_argumentation_ = True
                                hdoc[offset]._.vwp_argumentation_ = True
                                child._.vwp_argumentation_ = True
                    if (stancePredicate(hdoc[offset])
                        or hdoc[offset]._.vwp_information
                        or (hdoc[offset].dep_ != 'conj'
                            and (hdoc[offset]._.vwp_abstract
                                 or hdoc[offset]._.vwp_possession
                                 or hdoc[offset]._.vwp_cause
                                 or hdoc[offset]._.vwp_relation
                                 or hdoc[offset].dep_ in complements))):
                        if token._.is_academic \
                           or hdoc[offset]._.is_academic:
                            token._.vwp_argumentation_ = True
                            hdoc[offset]._.vwp_argumentation_ = True
                            for child in hdoc[offset].children:
                                if (stancePredicate(child)
                                    or child._.vwp_information
                                    or (child.dep_ != 'conj'
                                        and (child._.vwp_abstract
                                             or child._.vwp_possession
                                             or child._.vwp_cause
                                             or child._.vwp_relation))):
                                    child._.vwp_argumentation_ = True
                                    for grandchild in child.children:
                                        if grandchild.pos_ in ['DET', 'AUX'] \
                                           or grandchild.tag_ \
                                               in function_word_tags \
                                           or stance_adverb(grandchild) \
                                           or grandchild.lemma_ \
                                               in personal_or_indefinite_pronoun:
                                            grandchild._.vwp_argumentation_ \
                                                = True

        if token._.vwp_evaluation:
            for offset in getLinkedNodes(token):
                if (generalViewpointPredicate(hdoc[offset])
                    or hdoc[offset]._.vwp_information
                    or (hdoc[offset].dep_ != 'conj'
                        and (hdoc[offset]._.vwp_abstract
                             or hdoc[offset]._.vwp_possession
                             or hdoc[offset]._.vwp_cause
                             or hdoc[offset]._.vwp_relation))):
                    if token._.is_academic or hdoc[offset]._.is_academic:
                        token._.vwp_argumentation_ = True
                        hdoc[offset]._.vwp_argumentation_ = True

        if (token._.vwp_certainty
           or token._.vwp_probability
           or token._.vwp_qualification
           or token._.vwp_emphasis
           or token._.vwp_outcome
           or token._.vwp_surprise
           or token._.vwp_illocutionary
           or token._.vwp_generalization) \
           and token.dep_ == 'advmod' \
           and isRoot(token.head):
            token._.vwp_argumentation_ = True
            if token.head.is_stop:
                token.head._.vwp_argumentation_ = True

        if token.head._.vwp_argumentation_:
            if token.head.head.dep_ == 'prep':
                token.head.head._.vwp_argumentation_ = True

        if token._.vwp_argumentation_:
            if token.head.dep_ == 'prep':
                token.head._.vwp_argumentation_ = True

            if token.i + 1 < len(token.doc) \
               and token.nbor(1) is not None \
               and not token.nbor(1).lemma_ in core_temporal_preps \
               and ((token.nbor(1).pos_ in ['DET']
                     and token.nbor(1).lemma_ in quantifying_determiners)
                    or token.nbor(1).tag_ in function_word_tags
                    or stance_adverb(token.nbor(1))
                    or generalViewpointPredicate(token.nbor(1))
                    or token.nbor(1)._.vwp_information
                    or token.nbor(1)._.vwp_abstract
                    or token.nbor(1)._.vwp_possession
                    or token.nbor(1)._.vwp_cause
                    or token.nbor(1)._.vwp_relation
                    or token.nbor(1).dep_ in ['neg']):
                token.nbor(1)._.vwp_argumentation_ = True

            for child in token.children:
                if child.lemma_ not in core_temporal_preps \
                 and child.dep_ != 'conj' \
                 and (child.pos_ in ['DET', 'AUX']
                      or child.tag_ in function_word_tags
                      or stance_adverb(child)
                      or child.lemma_ in personal_or_indefinite_pronoun
                      or generalViewpointPredicate(child)
                      or child._.vwp_information
                      or (child.dep_ != 'conj'
                          and (child._.vwp_possession
                               or child._.vwp_relation
                               or child._.vwp_abstract
                               or child._.vwp_cause
                               or child.dep_ in ['neg']))):
                    child._.vwp_argumentation_ = True
                    for grandchild in child.children:
                        if grandchild.tag_ in function_word_tags \
                           or stance_adverb(grandchild):
                            grandchild._.vwp_argumentation_ = True
                            break
                    if token.tag_ == 'NOUN':
                        for grandchild in token.children:
                            if (stancePredicate(token)
                                or token._.vwp_information
                                or (grandchild.dep_ != 'conj'
                                    and (token._.vwp_abstract
                                         or token._.vwp_possession
                                         or token._.vwp_cause
                                         or token._.vwp_relation))):
                                grandchild._.vwp_argumentation_ = True

            if token._.vwp_argumentation_ \
               and token.tag_ in ['RB', 'MD', 'SCONJ']:
                for child in token.head.children:
                    if child.lemma_ not in core_temporal_preps \
                       and (child.tag_ in function_word_tags
                            or stance_adverb(child)
                            or child.lemma_ in personal_or_indefinite_pronoun
                            or generalViewpointPredicate(child)
                            or child._.vwp_information
                            or (child.dep_ != 'conj'
                                and (child._.vwp_abstract
                                     or child._.vwp_possession
                                     or child._.vwp_relation
                                     or child._.vwp_cause
                                     or child.dep_ in ['neg']))):
                        child._.vwp_argumentation_ = True
                        if (child.i + 1 < len(child.doc) and
                            (generalViewpointPredicate(child.nbor(1))
                             or child.nbor(1)._.vwp_information
                             or child.nbor(1)._.vwp_abstract
                             or child.nbor(1)._.vwp_possession
                             or child.nbor(1)._.vwp_cause
                             or child.nbor(1)._.vwp_relation
                             or child.nbor(1).dep_ in ['neg'])):
                            child.nbor(1)._.vwp_argumentation_ = True
                        for grandchild in child.children:
                            if generalViewpointPredicate(grandchild) \
                               or grandchild._.vwp_information \
                               or (grandchild.dep_ != 'conj'
                                   and (grandchild._.vwp_abstract
                                        or grandchild._.vwp_possession
                                        or grandchild._.vwp_relation
                                        or grandchild._.vwp_cause
                                        or grandchild.dep_ in ['neg'])):
                                grandchild._.vwp_argumentation_ = True

    def stance_perspective(self,
                           token,
                           hdoc,
                           referentID,
                           stance_markers,
                           pspans,
                           hdeps):

        # We mark words as argument-related if they involve
        # RELATIONSHIPS with other words from key categories
        # in our argument/perspective lexicon

        # If the subject and the verb disagree on viewpoint, correct
        # the verb to agree with the subject
        if token._.vwp_perspective_ is None \
           and token.dep_ in subject_dependencies:
            if token._.vwp_perspective_ != token.head._.vwp_perspective_:
                token.head._.vwp_perspective_ = token._.vwp_perspective_

        # Cleanup for stray cases where no viewpoint was assigned to ROOT
        if isRoot(token) \
           and token._.vwp_perspective_ is None:
            for child in token.children:
                if child._.vwp_perspective_ is not None:
                    token._.vwp_perspective_ = child._.vwp_perspective_
                    break

        # Cleanup -- prepositions should be assigned to the same
        # viewpoint as their head
        if token.dep_ == 'prep' \
           and token._.vwp_perspective_ is None:
            token._.vwp_perspective_ = token.head._.vwp_perspective_

        if token._.vwp_perspective_ is not None \
           and len(token._.vwp_perspective_) == 0:
            controller = self.getHeadDomain(token).i
            csubj = hdoc[controller]._.governing_subject_
            if (csubj is None or not hdoc[csubj]._.animate) \
               and getSubject(hdoc[controller]) is not None \
               and getSubject(hdoc[controller])._.animate:
                csubj = getSubject(hdoc[controller]).i
            if csubj is not None \
               and not hdoc[csubj]._.animate \
               and getSubject(hdoc[csubj]) is not None \
               and getSubject(hdoc[csubj])._.animate:
                csubj = getSubject(hdoc[csubj]).i

            if self.isHeadDomain(hdoc[controller]) \
               and len(getRoot(hdoc[controller])._.vwp_perspective_) == 0:
                controller = getRoot(hdoc[controller]).i
            if csubj is not None \
               and hdoc[csubj].lower_ in first_person_pronouns:
                domain = 'explicit_1'
                controller = csubj
            elif (csubj is not None
                  and hdoc[csubj].lower_
                  in second_person_pronouns):
                domain = 'explicit_2'
                controller = csubj
            else:
                domain = 'implicit'

            if domain not in pspans:
                pspans[domain] = {}
            if token._.transition \
               and token._.transition_category not in ['temporal',
                                                       'PARAGRAPH'] \
               and isRoot(token):
                if domain in ['explicit_1', 'explicit_2']:
                    if token.i not in pspans[domain]:
                        pspans[domain].append(token.i)
                        pspans[domain] = sorted(pspans[domain].copy())
                else:
                    if str(token.i) not in pspans[domain]:
                        pspans[domain][str(token.i)] = []
                    if token.i not in pspans[domain][str(token.i)]:
                        pspans[domain][str(token.i)].append(token.i)
                        pspans[domain][str(token.i)] = \
                            sorted(pspans[domain][str(token.i)].copy())
            else:
                if domain in ['explicit_1', 'explicit_2']:
                    if token.i not in pspans[domain]:
                        pspans[domain].append(token.i)
                        pspans[domain] = sorted(pspans[domain].copy())
                elif token.lower_ in first_person_pronouns:
                    if token.i not in pspans['explicit_1']:
                        pspans['explicit_1'].append(token.i)
                        pspans['explicit_1'] = \
                            sorted(pspans['explicit_1'].copy())
                elif token.lower_ in second_person_pronouns:
                    if token.i not in pspans['explicit_2']:
                        pspans['explicit_2'].append(token.i)
                        pspans['explicit_2'] = \
                            sorted(pspans['explicit_2'].copy())
                else:
                    if str(controller) not in pspans[domain]:
                        pspans[domain][str(controller)] = []
                    if token.i not in pspans[domain][str(controller)]:
                        pspans[domain][str(controller)].append(token.i)
                        pspans[domain][str(controller)] = \
                            sorted(pspans[domain][str(controller)].copy())
            deps = [child.dep_ for child in token.children]
            if token._.transition \
               and token._.transition_category not in ['temporal',
                                                       'PARAGRAPH'] \
               and isRoot(token):
                if domain in ['explicit_1', 'explicit_2']:
                    if token.i not in stance_markers[domain]:
                        stance_markers[domain].append(token.i)
                        stance_markers[domain] = sorted(
                            stance_markers[domain].copy())
                else:
                    if str(token.i) not in stance_markers[domain]:
                        stance_markers[domain][str(token.i)] = []
                    if token.i not in stance_markers[domain][str(token.i)]:
                        stance_markers[domain][str(token.i)].append(token.i)
                        stance_markers[domain][str(token.i)] = \
                            sorted(stance_markers[
                                domain][str(token.i)].copy())
            elif (token._.vwp_evaluation
                  or token._.vwp_hedge
                  or token._.subjectVerbInversion
                  or ((generalViewpointPredicate(token)
                       or (token._.vwp_raising
                           and token._.vwp_probability))
                      and (deps in clausal_complements
                           or deps in clausal_modifier_dependencies
                           or hdeps in clausal_complements
                           or hdeps in clausal_modifier_dependencies))
                  or (token._.vwp_character
                      and token._.governing_subject_ is not None
                      and hdoc[token._.governing_subject_]._.animate)
                  or (token._.vwp_evaluated_role
                      and token._.governing_subject_ is not None
                      and hdoc[token._.governing_subject_]._.animate)):
                if token.i not in stance_markers[domain]:
                    if token.dep_ == 'amod' \
                       and (token._.vwp_evaluation
                            or token._.vwp_hedge):
                        if domain in ['explicit_1', 'explicit_2']:
                            if token.i not in stance_markers[domain]:
                                stance_markers[domain].append(token.i)
                                stance_markers[domain] = \
                                    sorted(stance_markers[domain].copy())
                        else:
                            if str(token.i) not in stance_markers[domain]:
                                stance_markers[domain][str(token.i)] = []
                            if token.i not in \
                               stance_markers[domain][str(token.i)]:
                                stance_markers[domain][
                                    str(token.i)].append(token.i)
                                stance_markers[domain][str(token.i)] = \
                                    sorted(stance_markers[
                                        domain][str(token.i)].copy())
                    else:
                        if domain in ['explicit_1', 'explicit_2']:
                            if token.i not in stance_markers[domain]:
                                stance_markers[domain].append(token.i)
                                stance_markers[domain] = sorted(
                                    stance_markers[domain].copy())
                        else:
                            if str(controller) not in stance_markers[domain]:
                                stance_markers[domain][str(controller)] = []
                            if token.i not in stance_markers[
                               domain][str(controller)]:
                                stance_markers[domain][
                                    str(controller)].append(token.i)
                                stance_markers[domain][str(controller)] = \
                                    sorted(stance_markers[
                                         domain][str(controller)].copy())
        elif token is not None and token._.vwp_perspective_ is not None:
            for item in token._.vwp_perspective_:
                controller = hdoc[item]
                if controller.lower_ in first_person_pronouns \
                   and len(list(controller.children)) == 0 \
                   and not controller._.vwp_quoted:
                    if 'explicit_1' not in pspans:
                        pspans['explicit'] = []
                    if controller.i not in pspans['explicit_1'] \
                       and controller.i not in pspans['explicit_2']:
                        pspans['explicit_1'].append(controller.i)
                    for referent in referentID:
                        if referent not in pspans['explicit_1'] \
                           and referent not in pspans['explicit_2']:
                            pspans['explicit_1'].append(referent)
                            pspans['explicit_1'] = \
                                sorted(pspans['explicit_1'].copy())
                    if token._.vwp_evaluation \
                       or token._.vwp_hedge \
                       or token._.vwp_character \
                       or token._.vwp_evaluated_role:
                        if 'explicit_1' not in stance_markers:
                            stance_markers['explicit_1'] = []
                        if token.i not in stance_markers['explicit_1']:
                            stance_markers['explicit_1'].append(token.i)
                            stance_markers['explicit_1'] = \
                                sorted(stance_markers['explicit_1'].copy())

                elif (controller.lower_ in second_person_pronouns
                      and len(list(controller.children)) == 0
                      and not controller._.vwp_quoted):
                    if 'explicit_2' not in pspans:
                        pspans['explicit'] = []
                    if controller.i not in stance_markers['explicit_2'] \
                       and controller.i not in stance_markers['implicit']:
                        stance_markers['explicit_2'].append(controller.i)
                    for referent in referentID:
                        if referent not in pspans['explicit_2'] \
                           and referent not in pspans['implicit']:
                            pspans['explicit_2'].append(referent)
                            pspans['explicit_2'] = sorted(pspans['explicit_2'])
                    if token._.vwp_evaluation \
                       or token._.vwp_hedge \
                       or token._.vwp_character \
                       or token._.vwp_evaluated_role:
                        if 'explicit_2' not in stance_markers:
                            stance_markers['explicit_2'] = []
                        if token.i not in stance_markers['explicit_2']:
                            stance_markers['explicit_2'].append(token.i)
                            stance_markers['explicit_2'] = \
                                sorted(stance_markers['explicit_2'].copy())
                else:
                    if token.lower_ in first_person_pronouns:
                        if token.i not in pspans['explicit_1']:
                            pspans['explicit_1'].append(token.i)
                            pspans['explicit_1'] = sorted(
                                pspans['explicit_1'].copy())
                    if token.lower_ in second_person_pronouns:
                        if token.i not in pspans['explicit_2']:
                            pspans['explicit_2'].append(token.i)
                            pspans['explicit_2'] = sorted(
                                pspans['explicit_2'].copy())
                    else:
                        if token.lower_ in first_person_pronouns:
                            if token.i not in pspans['explicit_1']:
                                pspans['explicit_1'].append(token.i)
                                pspans['explicit_1'] = \
                                    sorted(pspans['explicit_1'].copy())
                            if 'explicit_1' not in stance_markers:
                                stance_markers['explicit_1'] = []
                            if token.i not in stance_markers['explicit_1']:
                                stance_markers['explicit_1'].append(token.i)
                                stance_markers['explicit_1'] = \
                                    sorted(stance_markers['explicit_1'].copy())
                        elif token.lower_ in second_person_pronouns:
                            if token.i not in pspans['explicit_2']:
                                pspans['explicit_2'].append(token.i)
                                pspans['explicit_2'] = \
                                    sorted(pspans[
                                        'explicit_2'].copy())
                            if 'explicit_2' not in stance_markers:
                                stance_markers['explicit_2'] = []
                            if token.i not in stance_markers['explicit_2']:
                                stance_markers['explicit_2'].append(token.i)
                                stance_markers['explicit_2'] = \
                                    sorted(stance_markers['explicit_2'].copy())
                        else:
                            if 'explicit_3' not in pspans:
                                pspans['explicit_3'] = {}
                            if int(item) not in pspans['explicit_3']:
                                pspans['explicit_3'][int(item)] = []
                            for referent in referentID:
                                if referent not in \
                                   pspans['explicit_3'][int(item)] \
                                   and referent not in pspans['explicit_1'] \
                                   and referent not in pspans['explicit_2']:
                                    pspans['explicit_3'][
                                        int(item)].append(referent)
                                    pspans['explicit_3'][int(item)] = \
                                        sorted(pspans[
                                            'explicit_3'][int(item)].copy())
                        if token._.vwp_evaluation \
                           or token._.vwp_hedge \
                           or token._.vwp_character \
                           or token._.vwp_evaluated_role:
                            if 'explicit_3' not in stance_markers:
                                stance_markers['explicit_3'] = {}
                            if int(item) not in stance_markers['explicit_3']:
                                stance_markers['explicit_3'][int(item)] = []
                            if token.i not in stance_markers['explicit_3'] \
                               and token.i not in \
                               stance_markers['explicit_1'] \
                               and token.i not in \
                               stance_markers['explicit_2']:
                                stance_markers['explicit_3'][
                                    int(item)].append(token.i)
                                stance_markers['explicit_3'][int(item)] = \
                                    sorted(stance_markers[
                                        'explicit_3'][int(item)].copy())
        return stance_markers, pspans

    def emotional_impact(self, token, hdoc, emotional_markers):
        if token._.vwp_emotional_impact \
           and getLogicalObject(token) is not None:
            controllers = ResolveReference(getLogicalObject(token), hdoc)
            charname = ''
            for i, controller in enumerate(controllers):
                if i == 0:
                    charname = hdoc[controller].text.capitalize()
                elif i == len(controllers)-1:
                    charname += ' and ' \
                        + hdoc[controller].text.capitalize()
                else:
                    charname += ', ' \
                        + hdoc[controller].text.capitalize()

            for controller in controllers:
                if getLogicalObject(token)._.animate:
                    if hdoc[controller].lower_ in first_person_pronouns \
                       and len(list(
                           hdoc[
                               controller].children)) == 0 \
                       and not hdoc[controller]._.vwp_quoted:
                        if token.i not in emotional_markers['explicit_1']:
                            emotional_markers['explicit_1'].append(token.i)
                    elif (hdoc[controller].lower_
                          in second_person_pronouns
                          and len(list(hdoc[controller].children)) == 0
                          and not hdoc[controller]._.vwp_quoted):
                        if token.i not in emotional_markers['explicit_2']:
                            emotional_markers['explicit_2'].append(token.i)
                    else:
                        if controller not in \
                           emotional_markers['explicit_3']:
                            emotional_markers[
                                'explicit_3'][charname] = []
                            if token.i not in \
                               emotional_markers[
                                   'explicit_3'][charname]:
                                emotional_markers[
                                    'explicit_3'][
                                        charname].append(token.i)
                        else:
                            if token.i not in \
                               emotional_markers[
                                   'explicit_3'][charname]:
                                if token.i not in \
                                   emotional_markers[
                                       'explicit_3'][charname]:
                                    emotional_markers[
                                         'explicit_3'][
                                             charname].append(token.i)
        return emotional_markers

    def character_traits(self, token, hdoc, referentID, character_markers):
        if (token.pos_ == 'NOUN' and token._.vwp_evaluated_role) \
           or token._.vwp_character:
            if token._.has_governing_subject:
                controllers = ResolveReference(
                    hdoc[token._.governing_subject], hdoc)
                charname = ''
                for i, controller in enumerate(controllers):
                    if i == 0:
                        charname = hdoc[controller].text.capitalize()
                    elif i == len(controllers)-1:
                        charname += ' and ' \
                            + hdoc[controller].text.capitalize()
                    else:
                        charname += ', ' + hdoc[controller].text.capitalize()
                for controller in controllers:
                    if hdoc[controller]._.animate:
                        if hdoc[controller].lower_ in \
                           first_person_pronouns \
                           and len(list(
                               hdoc[controller].children)) == 0:
                            if referentID not in \
                               character_markers['explicit_1']:
                                for referent in referentID:
                                    if referent not in \
                                       character_markers[
                                           'explicit_1']:
                                        character_markers[
                                            'explicit_1'].append(
                                                referent)
                        elif hdoc[controller].lower_ \
                            in second_person_pronouns \
                            and len(list(
                                hdoc[controller].children)) == 0:
                            if referentID not in \
                               character_markers['explicit_2']:
                                for referent in referentID:
                                    if referent not in \
                                       character_markers['explicit_2']:
                                        character_markers[
                                            'explicit_2'].append(
                                                referent)
                        else:
                            if charname not in \
                               character_markers['explicit_3']:
                                character_markers[
                                    'explicit_3'][charname] = []
                                for referent in referentID:
                                    if referent not in controllers:
                                        if referent not in \
                                           character_markers[
                                                'explicit_3'][charname]:
                                            character_markers[
                                                'explicit_3'][
                                                    charname].append(
                                                        referent)
                            else:
                                if referentID not in \
                                   character_markers[
                                       'explicit_3'][charname]:
                                    for referent in referentID:
                                        if referent \
                                           not in controllers:
                                            if referent not in \
                                               character_markers[
                                                   'explicit_3'][charname]:
                                                character_markers[
                                                    'explicit_3'][
                                                        charname].append(
                                                            referent)
                    break
        return character_markers

    def emotion_predicates(self, token, hdoc, emotional_markers):
        if token._.vwp_emotion:
            if token._.has_governing_subject:
                controllers = ResolveReference(
                    hdoc[token._.governing_subject], hdoc)
                for i, controller in enumerate(controllers):
                    if i == 0:
                        charname = \
                            hdoc[controller].text.capitalize()
                    elif i == len(controllers)-1:
                        charname += ' and '\
                            + hdoc[controller].text.capitalize()
                    else:
                        charname += ', ' + hdoc[controller].text.capitalize()
                for controller in controllers:
                    if hdoc[controller]._.animate:
                        if hdoc[controller].lower_ \
                            in first_person_pronouns \
                            and len(list(
                              hdoc[controller].children)) == 0:
                            if token.i not in \
                               emotional_markers['explicit_1'] \
                                and not \
                                    hdoc[controller]._.vwp_quoted:
                                if token.i not in \
                                   emotional_markers['explicit_1']:
                                    emotional_markers[
                                        'explicit_1'].append(token.i)
                        elif hdoc[controller].lower_ \
                            in second_person_pronouns \
                            and len(list(
                              hdoc[controller].children)) == 0:
                            if token.i not in \
                               emotional_markers['explicit_2'] \
                               and not hdoc[controller]._.vwp_quoted:
                                if token.i not in \
                                   emotional_markers['explicit_2']:
                                    emotional_markers[
                                        'explicit_2'].append(token.i)
                        else:
                            if charname not in \
                               emotional_markers['explicit_3']:
                                emotional_markers[
                                    'explicit_3'][charname] = []
                                if token.i not in \
                                   emotional_markers[
                                       'explicit_3'][charname]:
                                    emotional_markers[
                                        'explicit_3'][
                                            charname].append(token.i)
                            else:
                                if token.i not in emotional_markers[
                                   'explicit_3'][charname]:
                                    if token.i not in emotional_markers[
                                       'explicit_3'][
                                           charname]:
                                        emotional_markers[
                                            'explicit_3'][
                                                charname].append(token.i)
        return emotional_markers

    def propDomain(self, token, controller, hdoc):
        if controller is None:
            controller = token._.governing_subject
        dom = self.getHeadDomain(token)
        if controller is not None:
            ctrlr = hdoc[controller]
            while (ctrlr.dep_ is not None
                   and ctrlr != ctrlr.head
                   and not isRoot(ctrlr)
                   and ctrlr.pos_ != 'VERB'):
                ctrlr = ctrlr.head
            if ctrlr.i != dom.i:
                dom = ctrlr.head
        start = list(dom.subtree)[0].i
        end = list(dom.subtree)[len(list(dom.subtree))-1].i
        return ([start, end], controller, token.i)

    def cleanup_propositional_attitudes(self, propositional_attitudes, hdoc, name):
        for item in propositional_attitudes['implicit_3']:
            if item in propositional_attitudes['implicit']:
                propositional_attitudes['implicit'].remove(item)
        for item in propositional_attitudes['explicit_1']:
            if item in propositional_attitudes['implicit']:
                propositional_attitudes['implicit'].remove(item)
            if item in propositional_attitudes['implicit_3']:
                propositional_attitudes['implicit_3'].remove(item)
        for item in propositional_attitudes['explicit_2']:
            if item in propositional_attitudes['implicit']:
                propositional_attitudes['implicit'].remove(item)
            if item in propositional_attitudes['implicit_3']:
                propositional_attitudes['implicit_3'].remove(item)
        for domain in propositional_attitudes['explicit_3']:
            for item in propositional_attitudes['explicit_3'][domain]:
                if item in propositional_attitudes['implicit']:
                    propositional_attitudes['implicit'].remove(item)
                if item in propositional_attitudes['implicit_3']:
                    propositional_attitudes['implicit_3'].remove(item)

        # register which tokens are part of claims or discussions
        for item in propositional_attitudes['implicit']:
            if isinstance(item, tuple):
                for offset in range(item[0][0], item[0][1]):
                    hdoc[offset]._.vwp_claim_ = True
        for item in propositional_attitudes['explicit_1']:
            if isinstance(item, tuple):
                for offset in range(item[0][0], item[0][1]):
                    hdoc[offset]._.vwp_claim_ = True
        for item in propositional_attitudes['explicit_2']:
            if isinstance(item, tuple):
                for offset in range(item[0][0], item[0][1]):
                    hdoc[offset]._.vwp_claim_ = True
        for item in propositional_attitudes['implicit_3']:
            if isinstance(item, tuple):
                for offset in range(item[0][0], item[0][1]):
                    hdoc[offset]._.vwp_discussion = True
        for domain in propositional_attitudes['explicit_3']:
            if isinstance(item, tuple):
                for item in propositional_attitudes['explicit_3'][domain]:
                    for offset in range(item[0][0], item[0][1]):
                        hdoc[offset]._.vwp_discussion_ = True

        # Now let's clean things up to put propositional attitudes
        # in the format we're standardizing to for sentence scale ranges
        reformatted = []
        for item in propositional_attitudes['implicit']:
            if type(item) == str \
               and type(propositional_attitudes['implicit'])==dict:
                values = propositional_attitudes['implicit'][item]
                for value in values:
                    entry = newSpanEntry(name,
                                         int(value),
                                         int(value),
                                         hdoc,
                                         'SELF')
                    reformatted.append(entry)
                continue
            elif type(item) in [str,int]:
                entry = newSpanEntry(name,
                                     int(item),
                                     int(item),
                                     hdoc,
                                     'SELF')
            else:
                [[left, right], controller, proposition] = item
                entry = newSpanEntry(name,
                                     left,
                                     right,
                                     hdoc,
                                     'implicit')
            reformatted.append(entry)
        for item in propositional_attitudes['explicit_1']:
            if type(item) in [str,int]:
                entry = newSpanEntry(name,
                                     int(item),
                                     int(item),
                                     hdoc,
                                     'SELF')
            else:
                [[left, right], controller, proposition] = item
                entry = newSpanEntry(name,
                                     left,
                                     right,
                                     hdoc,
                                     'SELF')
            reformatted.append(entry)
        for item in propositional_attitudes['explicit_2']:
            if type(item) in [str,int]:
                entry = newSpanEntry(name,
                                     int(item),
                                     int(item),
                                     hdoc,
                                     'AUDIENCE')
            else:
                [[left, right], controller, proposition] = item
                entry = newSpanEntry(name,
                                     left,
                                     right,
                                     hdoc,
                                     'AUDIENCE')
            reformatted.append(entry)
        for domain in propositional_attitudes['explicit_3']:
            for item in propositional_attitudes['explicit_3'][domain]:
                if type(item) in [str,int]:
                    entry = newSpanEntry(name,
                                         int(item),
                                         int(item),
                                         hdoc,
                                         domain)
                else:
                    [[left, right], controller, proposition] = item
                    entry = newSpanEntry(name,
                                         left,
                                         right,
                                         hdoc,
                                         controller)
                reformatted.append(entry)
        return reformatted

    def vwp_character_traits(self, hdoc):
        character_markers = {}
        character_markers['implicit'] = []
        character_markers['implicit_1'] = []
        character_markers['implicit_3'] = []
        character_markers['explicit_1'] = []
        character_markers['explicit_2'] = []
        character_markers['explicit_3'] = {}
        for token in hdoc:
            referentID = ResolveReference(token, hdoc)
            if token.i not in referentID:
                referentID.append(token.i)
            if token._.has_governing_subject:
                character_markers = \
                    self.character_traits(token,
                                          hdoc,
                                          referentID,
                                          character_markers)
        return self.cleanup_propositional_attitudes(
           character_markers, hdoc, 'character_traits')

    def vwp_emotion_states(self, hdoc):
        '''
           Identification of emotional predicates predicates of a particular
           viewpoint-holding nominal
        '''
        emotional_markers = {}
        emotional_markers['implicit'] = []
        emotional_markers['implicit_1'] = []
        emotional_markers['implicit_3'] = []
        emotional_markers['explicit_1'] = []
        emotional_markers['explicit_2'] = []
        emotional_markers['explicit_3'] = {}
        for token in hdoc:
            emotional_markers = \
                self.emotional_impact(token,
                                      hdoc,
                                      emotional_markers)
            emotional_markers = \
                self.emotion_predicates(token,
                                        hdoc,
                                        emotional_markers)
        return self.cleanup_propositional_attitudes(
           emotional_markers, hdoc, 'emotion_states')


    def vwp_propositional_attitudes(self, hdoc):
        '''
           Identification of propositional attitude predicates associated
           with specific predicates. E.g., believe or think in 'I believe
           that this is true', or 'John thinks we are on the right track'.
        '''
        if hdoc._.propositional_attitudes_ is not None:
            return hdoc._.propositional_attitudes_
        
        propositional_attitudes = {}
        propositional_attitudes['implicit'] = []
        propositional_attitudes['implicit_3'] = []
        propositional_attitudes['explicit_1'] = []
        propositional_attitudes['explicit_2'] = []
        propositional_attitudes['explicit_3'] = {}
        for token in hdoc:
            hdeps = [child.dep_ for child in token.head.children]
            propositional_attitudes = \
                self.propositional_attitudes(
                    token, hdoc, propositional_attitudes, hdeps)
        hdoc._.propositional_attitudes_ = self.cleanup_propositional_attitudes(
           propositional_attitudes, hdoc, 'propositional attitudes')
        return hdoc._.propositional_attitudes_

    def vwp_perspective_spans(self, hdoc):
        self.markPerspectiveSpan(hdoc)
        return hdoc._.vwp_perspective_spans_

    def vwp_stance_markers(self, hdoc):
        self.markPerspectiveSpan(hdoc)
        return hdoc._.vwp_stance_markers_

    def vwp_attribution(self, token):
        '''
           Store information about whether a predicate
           is a claim predicate
        '''
        self.vwp_direct_speech(token.doc)
        return token._.vwp_attribution_

    def vwp_source(self, token):
        '''
           Store information about whether a predicate
           is a claim predicate
        '''
        self.vwp_direct_speech(token.doc)
        return token._.vwp_source_

    def vwp_cite(self, token):
        '''
           Store information about whether a predicate
           is a claim predicate
        '''
        self.vwp_direct_speech(token.doc)
        return token._.vwp_cite_


    def vwp_claim(self, token):
        '''
           Store information about whether a predicate
           is a claim predicate
        '''
        self.vwp_propositional_attitudes(token.doc)
        return token._.vwp_claim_

    def vwp_discussion(self, token):
        '''
           Store information about whether a predicate
           is a discussion predicate (propositional
           attitude predicate for a third person point of
           view, e.g., his statement that he supported me
        '''
        self.vwp_propositional_attitudes(token.doc)
        return token._.vwp_discussion_

    def propositional_attitudes(self,
                                token,
                                hdoc,
                                propositional_attitudes,
                                hdeps):
        '''
           Calculate whether tokens define a propositional attitude
           (claim + proposition)
        '''
        if ((coreViewpointPredicate(token.head)
             or token.head._.vwp_evaluation
             or (token._.transition
                 and token._.transition_category is not None
                 and token._.transition_category not in ['temporal',
                                                         'PARAGRAPH'])
            or any([(coreViewpointPredicate(child)
                     or (child._.transition
                         and child._.transition_category is not None
                         and child._.transition_category not in ['temporal',
                                                                 'PARAGRAPH'])
                    or child._.vwp_evaluation)
                   for child in token.head.children
                   if child.dep_ in adjectival_complement_dependencies])
            or (token.head.lower_ in be_verbs
                and token.head._.governing_subject is not None
                and hdoc[
                    token.head._.governing_subject]._.vwp_evaluation)
            or (token.head.lower_ in be_verbs)
                and 'attr' not in hdeps
                and 'acomp' not in hdeps)
           and ((token.dep_ in ['ccomp', 'csubjpass', 'acl', 'oprd']
                 and tensed_clause(token))
                or token.dep_ in 'relcl' and token.head.dep_ in ['nsubj',
                                                                 'nsubjpass',
                                                                 'attr']
                or (token.dep_ in ['ccomp', 'oprd']
                    and ('dobj' in hdeps
                         or 'nsubjpass' in hdeps))
                or (token.dep_ == 'advcl'
                    and 'for' in [child.lemma_ for child in token.children
                                  if child.dep_ == 'mark'])
                or (token.dep_ == 'prep'
                    and 'pcomp' in [child.dep_ for child in token.children
                                    if tensed_clause(child)]))
           and ((getSubject(token.head) is None
                 and token.head._.governing_subject is None)
                or (token.head._.governing_subject is not None
                    and hdoc[token.head._.governing_subject].lemma_ == 'it'
                    and (token.head._.vwp_raising
                         or token.head._.vwp_tough
                         or token.head.lower_
                         in be_verbs))
                or (token.head._.governing_subject is not None
                    and (hdoc[token.head._.governing_subject]._.animate
                         or hdoc[token.head._.governing_subject
                                 ]._.vwp_sourcetext
                         or coreViewpointPredicate(hdoc[
                             token.head._.governing_subject])
                         or (hdoc[token.head._.governing_subject
                                  ]._.transition
                             and hdoc[
                                      token.head._.governing_subject
                                      ]._.transition_category
                             is not None
                             and hdoc[
                                      token.head._.governing_subject
                                      ]._.transition_category
                             not in ['temporal', 'PARAGRAPH'])
                         or hdoc[token.head._.governing_subject
                                 ]._.vwp_evaluation)))) \
                or (token.dep_ in object_predicate_dependencies
                    and (coreViewpointPredicate(token)
                         or (token._.transition
                             and token._.transition_category is not None
                             and token._.transition_category
                             not in ['temporal', 'PARAGRAPH'])
                         or token.head._.vwp_evaluation)
                    and token.head._.governing_subject is not None
                    and (hdoc[token.head._.governing_subject].dep_ == 'csubj'
                         or hdoc[token.head._.governing_subject
                                 ].tag_ == '_SP')):
            domHead = self.propDomain(token.head,
                                      token.head._.governing_subject,
                                      hdoc)
            if token.head._.governing_subject is None:
                if token.head.pos_ == 'VERB':
                    # Imperatives
                    if isRoot(token) \
                       and token.head.lemma_ == token.head.lower_ \
                       and not all(['Tense=' in str(child.morph)
                                    for child in token.head.children]):
                        if domHead not in \
                           propositional_attitudes['explicit_2']:
                            propositional_attitudes[
                                'explicit_2'].append(domHead)
                        elif is_definite_nominal(token.head):
                            if domHead not in \
                               propositional_attitudes['implicit_3']:
                                propositional_attitudes[
                                    'implicit_3'].append(domHead)
                    # Controlled verbs
                    else:
                        if is_definite_nominal(token.head):
                            if domHead not in \
                               propositional_attitudes['implicit_3']:
                                propositional_attitudes[
                                    'implicit_3'].append(domHead)
                            elif token.head.pos_ == 'NOUN':
                                if domHead not in \
                                   propositional_attitudes['implicit_3']:
                                    propositional_attitudes[
                                        'implicit_3'].append(domHead)
                        else:
                            if domHead not in \
                               propositional_attitudes['implicit']:
                                propositional_attitudes[
                                    'implicit'].append(domHead)
                # nominals w/o subject
                elif token.head.pos_ in ['ADJ']:
                    if domHead not in propositional_attitudes['implicit']:
                        propositional_attitudes['implicit'].append(domHead)
                elif token.head.pos_ in ['NOUN']:
                    if is_definite_nominal(token.head):
                        if domHead not in \
                           propositional_attitudes['implicit_3']:
                            propositional_attitudes[
                                'implicit_3'].append(domHead)
                    else:
                        if domHead not in propositional_attitudes['implicit']:
                            propositional_attitudes['implicit'].append(domHead)
            else:
                controllers = ResolveReference(
                    hdoc[token.head._.governing_subject], hdoc)
                for controller in controllers:
                    domHead = self.propDomain(token.head, controller, hdoc)
                    if (hdoc[
                        token.head._.governing_subject].lemma_ == 'it'
                        and (token.head._.vwp_raising
                             or token.head._.vwp_tough
                             or token.head._.vwp_evaluation
                             or any([child._.vwp_evaluation
                                     for child in token.head.children]))):
                        if token.head.pos_ in ['NOUN']:
                            if is_definite_nominal(token.head):
                                if token.head.i not in \
                                   propositional_attitudes['implicit_3']:
                                    propositional_attitudes[
                                        'implicit_3'].append(token.head.i)
                            else:
                                if token.head.i not in \
                                   propositional_attitudes['implicit']:
                                    propositional_attitudes[
                                        'implicit'].append(domHead)
                        else:
                            if token.head.i not in \
                               propositional_attitudes['implicit']:
                                propositional_attitudes[
                                    'implicit'].append(domHead)
                    elif (hdoc[controller].lower_
                          in first_person_pronouns
                          and len(list(hdoc[controller].children)) == 0):
                        if token.head.i not in \
                           propositional_attitudes['explicit_1'] \
                           and not hdoc[controller]._.vwp_quoted:
                            propositional_attitudes[
                                'explicit_1'].append(domHead)
                    elif (hdoc[controller].lower_
                          in second_person_pronouns
                          and len(list(hdoc[controller].children)) == 0):
                        if domHead not in propositional_attitudes[
                           'explicit_2'] \
                           and not hdoc[controller]._.vwp_quoted:
                            propositional_attitudes[
                                'explicit_2'].append(domHead)
                    else:
                        if coreViewpointPredicate(hdoc[controller]) \
                           or (hdoc[controller]._.transition
                               and (hdoc[controller]._.transition_category
                                    is not None)
                               and (hdoc[controller]._.transition_category
                                    not in ['temporal', 'PARAGRAPH'])) \
                           or hdoc[controller]._.vwp_evaluation:
                            if is_definite_nominal(token.head):
                                if domHead not in \
                                   propositional_attitudes['implicit_3']:
                                    propositional_attitudes[
                                        'implicit_3'].append(domHead)
                            else:
                                if domHead not in \
                                   propositional_attitudes['implicit']:
                                    propositional_attitudes[
                                        'implicit'].append(domHead)
                        else:
                            if controller not in \
                               propositional_attitudes['explicit_3']:
                                propositional_attitudes[
                                    'explicit_3'][controller] = []
                                if domHead not in \
                                   propositional_attitudes[
                                       'explicit_3'][controller]:
                                    propositional_attitudes[
                                        'explicit_3'][
                                            controller].append(domHead)
                            else:
                                if domHead not in \
                                   propositional_attitudes[
                                       'explicit_3'][controller]:
                                    propositional_attitudes[
                                        'explicit_3'][
                                            controller].append(
                                                domHead)

        elif (token.dep_ == 'amod'
              and token.head.dep_ in object_predicate_dependencies
              and (coreViewpointPredicate(token)
                   or (token._.transition
                       and token._.transition_category is not None
                       and token._.transition_category
                       not in ['temporal', 'PARAGRAPH'])
                   or token._.vwp_evaluation)
              and token._.governing_subject is not None
              and (hdoc[token._.governing_subject].dep_ == 'csubj'
                   or hdoc[token._.governing_subject].tag_ == '_SP')):

            controllers = ResolveReference(
                hdoc[token.head.head._.governing_subject], hdoc)
            for controller in controllers:
                domHead = self.propDomain(token.head.head, controller, hdoc)
                if (hdoc[token.head.head._.governing_subject].lemma_ == 'it'
                   and (token.head.head._.vwp_raising
                        or token.head.head._.vwp_tough
                        or token.head.head._.vwp_evaluation)):
                    if token.head.head.pos_ == 'NOUN':
                        if is_definite_nominal(token.head.head):
                            if domHead not in \
                               propositional_attitudes['implicit_3']:
                                propositional_attitudes[
                                    'implicit_3'].append(domHead)
                        else:
                            if domHead not in \
                               propositional_attitudes['implicit']:
                                propositional_attitudes[
                                    'implicit'].append(domHead)
                    else:
                        if domHead not in propositional_attitudes['implicit']:
                            propositional_attitudes[
                                'implicit'].append(domHead)
                elif (hdoc[controller].lower_
                      in first_person_pronouns
                      and len(list(hdoc[controller].children)) == 0):
                    if domHead not in propositional_attitudes['explicit_1'] \
                       and not hdoc[controller]._.vwp_quoted:
                        propositional_attitudes['explicit_1'].append(domHead)
                elif (hdoc[controller].lower_ in second_person_pronouns
                      and len(list(hdoc[controller].children)) == 0):
                    if domHead not in propositional_attitudes['explicit_2'] \
                       and not hdoc[controller]._.vwp_quoted:
                        propositional_attitudes['explicit_2'].append(domHead)
                else:
                    if coreViewpointPredicate(token) \
                       or (hdoc[controller]._.transition
                           and hdoc[controller]._.transition_category
                           is not None
                           and hdoc[controller]._.transition_category
                           not in ['temporal', 'PARAGRAPH']) \
                       or hdoc[controller]._.vwp_evaluation:
                        if is_definite_nominal(token.head.head):
                            if domHead not in \
                               propositional_attitudes['implicit_3']:
                                propositional_attitudes[
                                    'implicit_3'].append(domHead)
                        else:
                            if domHead not in \
                               propositional_attitudes['implicit']:
                                propositional_attitudes[
                                    'implicit'].append(domHead)
                    else:
                        if controller not in \
                           propositional_attitudes['explicit_3']:
                            propositional_attitudes[
                                'explicit_3'][controller] = []
                            if domHead not in propositional_attitudes[
                               'explicit_3'][controller]:
                                propositional_attitudes[
                                    'explicit_3'][controller].append(domHead)
                        else:
                            if domHead not in propositional_attitudes[
                               'explicit_3'][controller]:
                                propositional_attitudes[
                                    'explicit_3'][
                                        controller].append(domHead)
        elif (token.dep_ == 'acl'
              and token.head.head.lemma_ == 'for'
              and token.head.head.head.pos_ == 'ADJ'
              and token.head.head.head._.vwp_emotion
              and token.head.head.head._.governing_subject
              is not None):
            controllers = ResolveReference(
                hdoc[token.head.head.head._.governing_subject], hdoc)
            for controller in controllers:
                domHead = self.propDomain(
                    token.head.head.head, controller, hdoc)
                if (hdoc[
                    token.head.head.head._.governing_subject].lemma_ == 'it'
                    and (token.head.head.head._.vwp_raising
                         or token.head.head.head._.vwp_tough
                         or token.head.head.head._.vwp_evaluation)):
                    if token.head.head.head.pos_ == 'NOUN':
                        if is_definite_nominal(token.head.head.head):
                            if domHead not in propositional_attitudes[
                               'implicit_3']:
                                propositional_attitudes[
                                    'implicit_3'].append(domHead)
                        else:
                            if domHead not in propositional_attitudes[
                               'implicit']:
                                propositional_attitudes[
                                    'implicit'].append(domHead)
                    else:
                        if domHead not in propositional_attitudes['implicit']:
                            propositional_attitudes['implicit'].append(domHead)
                elif (hdoc[controller].lower_ in first_person_pronouns
                      and len(list(hdoc[controller].children)) == 0):
                    if domHead not in propositional_attitudes['explicit_1'] \
                       and not hdoc[controller]._.vwp_quoted:
                        propositional_attitudes['explicit_1'].append(domHead)
                elif (hdoc[controller].lower_ in second_person_pronouns
                      and len(list(hdoc[controller].children)) == 0):
                    if domHead not in propositional_attitudes['explicit_2'] \
                       and not hdoc[controller]._.vwp_quoted:
                        propositional_attitudes['explicit_2'].append(domHead)
                else:
                    if coreViewpointPredicate(hdoc[controller]) \
                       or (hdoc[controller]._.transition
                           and hdoc[controller]._.transition_category
                           is not None
                           and hdoc[controller]._.transition_category
                           not in ['temporal', 'PARAGRAPH']) \
                       or hdoc[controller]._.vwp_evaluation:
                        if is_definite_nominal(token.head.head.head):
                            if domHead not in \
                               propositional_attitudes['implicit_3']:
                                propositional_attitudes[
                                    'implicit_3'].append(domHead)
                        else:
                            if domHead not in \
                               propositional_attitudes['implicit']:
                                propositional_attitudes[
                                    'implicit'].append(domHead)
                    else:
                        if controller not in \
                           propositional_attitudes['explicit_3']:
                            propositional_attitudes[
                                'explicit_3'][controller] = []
                            if domHead not in \
                               propositional_attitudes[
                                   'explicit_3'][controller]:
                                propositional_attitudes[
                                    'explicit_3'][controller].append(domHead)
                        else:
                            if domHead not in propositional_attitudes[
                               'explicit_3'][controller]:
                                propositional_attitudes[
                                    'explicit_3'][
                                        controller].append(domHead)
        elif (coreViewpointPredicate(token)
              or (token._.vwp_evaluation
                  and not token._.vwp_manner)
              or token.dep_ == 'neg'
              or token.lemma_ == 'no'
              or (token._.transition
                  and token._.transition_category is not None
                  and token._.transition_category
                  not in ['temporal', 'PARAGRAPH'])
              or token._.vwp_generalization
              or token._.vwp_illocutionary
              or token._.vwp_probability
              or token._.vwp_reservation
              or token._.vwp_emphasis):
            domHead = self.propDomain(token.head.head, None, hdoc)
            if token.dep_ in adjectival_complement_dependencies \
               and getSubject(token.head) is not None \
               and getSubject(token.head).dep_ == 'csubj':
                if token._.governing_subject is not None \
                   and hdoc[token._.governing_subject]._.animate:
                    domain = hdoc[token._.governing_subject].i
                    domHead = self.propDomain(token.head.head.head,
                                              domain,
                                              hdoc)
                    if hdoc[token._.governing_subject].lower_ \
                       in first_person_pronouns:
                        if domHead not in propositional_attitudes[
                           'explicit_1']:
                            propositional_attitudes[
                                'explicit_1'].append(domHead)
                    elif (hdoc[token._.governing_subject].lower_
                          in second_person_pronouns):
                        if domHead not in propositional_attitudes[
                           'explicit_2']:
                            propositional_attitudes[
                                'explicit_2'].append(domHead)
                    else:
                        if domain not in propositional_attitudes['explicit_3']:
                            propositional_attitudes[
                                'explicit_3'][domain] = []
                        if domHead not in propositional_attitudes[
                           'explicit_3'][domain]:
                            propositional_attitudes[
                                'explicit_3'][domain].append(domHead)
                elif token._.governing_subject is not None:
                    domain = hdoc[token._.governing_subject].i
                    domHead = self.propDomain(token.head.head, domain, hdoc)
                    if domHead not in propositional_attitudes['implicit']:
                        propositional_attitudes['implicit'].append(domHead)
            if token.pos_ == 'ADV' \
               and token.head.pos_ in ['VERB', 'AUX'] \
               and (self.getHeadDomain(token).dep_ is None
                    or isRoot(self.getHeadDomain(token))
                    or self.getHeadDomain(token).head.tag_ == '_SP'):
                if domHead not in propositional_attitudes['implicit']:
                    propositional_attitudes['implicit'].append(domHead)
            elif (token.pos_ in 'ADV'
                  and token.head.dep_ in adjectival_complement_dependencies
                  and token.head.head.pos_ in ['VERB', 'AUX']
                  and (self.getHeadDomain(token.head.head).dep_ is None
                       or isRoot(self.getHeadDomain(token.head.head)))):
                if domHead not in propositional_attitudes['implicit']:
                    propositional_attitudes['implicit'].append(domHead)

            elif (token.dep_ in ['ccomp',
                                 'csubj',
                                 'csubjpass',
                                 'acl']
                  and token.head._.vwp_evaluation):
                if domHead not in propositional_attitudes['implicit']:
                    propositional_attitudes['implicit'].append(domHead)

            elif (token.dep_ == 'neg'
                  and (self.getHeadDomain(token).dep_ is None
                       or isRoot(self.getHeadDomain(token)))):
                if domHead not in propositional_attitudes['implicit']:
                    propositional_attitudes['implicit'].append(domHead)
            elif (token.dep_ == 'det'
                  and token.lemma_ == 'no'
                  and isRoot(self.getHeadDomain(token))):
                if domHead not in propositional_attitudes['implicit']:
                    propositional_attitudes['implicit'].append(domHead)
            elif (token.dep_ in auxiliary_dependencies
                  and token._.vwp_evaluation
                  and token.head.pos_ in ['VERB', 'AUX']
                  and isRoot(self.getHeadDomain(token))):
                if domHead not in propositional_attitudes['implicit']:
                    propositional_attitudes['implicit'].append(domHead)
            elif token._.governing_subject is not None:
                match = False
                for item in hdoc[token._.governing_subject].subtree:
                    if item.dep_ in ['ccomp', 'acl']:
                        match = True
                        break
                if match:
                    domHead = self.propDomain(token.head.head.head, None, hdoc)
                    if (hdoc[token._.governing_subject].lemma_ == 'it'
                        and (token._.vwp_raising
                             or token._.vwp_tough
                             or token._.vwp_evaluation)):
                        if token.head.head.head.pos_ == 'NOUN':
                            if domHead not in \
                               propositional_attitudes['implicit_3']:
                                propositional_attitudes[
                                   'implicit_3'].append(domHead)
                            else:
                                if domHead not in \
                                   propositional_attitudes['implicit']:
                                    propositional_attitudes[
                                       'implicit'].append(domHead)
                        else:
                            if domHead not in \
                               propositional_attitudes['implicit']:
                                propositional_attitudes[
                                    'implicit'].append(domHead)
                    else:
                        if token.head.head.head._.governing_subject \
                           is not None:
                            controllers = ResolveReference(
                                hdoc[token.head.head.head._.governing_subject
                                     ], hdoc)
                            for controller in controllers:
                                if hdoc[controller].lower_ \
                                   in first_person_pronouns \
                                   and len(list(
                                      hdoc[controller].children)) == 0:
                                    if domHead not in \
                                       propositional_attitudes['explicit_1'] \
                                       and not hdoc[controller]._.vwp_quoted:
                                        propositional_attitudes[
                                            'explicit_1'].append(domHead)
                                elif (hdoc[controller].lower_
                                      in second_person_pronouns
                                      and len(list(
                                          hdoc[controller].children)) == 0):
                                    if domHead not in \
                                        propositional_attitudes['explicit_2'] \
                                       and not hdoc[controller]._.vwp_quoted:
                                        propositional_attitudes[
                                            'explicit_2'].append(domHead)
                                else:
                                    if domHead not in \
                                       propositional_attitudes['implicit_3']:
                                        propositional_attitudes[
                                            'implicit_3'].append(domHead)
        if is_definite_nominal(token) \
           and isRoot(self.getHeadDomain(token)) \
           and (coreViewpointPredicate(token)
                or (token._.transition
                    and token._.transition_category is not None
                    and token._.transition_category
                    not in ['temporal', 'PARAGRAPH'])
                or token._.vwp_evaluation):
            rt = self.propDomain(token.head.head, None, hdoc)
            if tensed_clause(self.getHeadDomain(token)
               or self.getHeadDomain(token)._.pos_ != 'VERB'):
                if rt not in propositional_attitudes['implicit_3'] \
                   and rt not in propositional_attitudes['implicit']:
                    propositional_attitudes['implicit_3'].append(rt)
            else:
                if rt not in propositional_attitudes['explicit_2'] \
                   and rt not in propositional_attitudes['explicit_2']:
                    propositional_attitudes['explicit_2'].append(rt)

        return propositional_attitudes

    def vwp_social_awareness(self, doc):
        '''
            Sentence structures that indicate that one viewpoint-taker
            is aware of another viewpoint (I think that he knows ...)
        '''
        theory_of_mind_sentences = []
        for token in doc:
            theory_of_mind_sentences =\
                self.theory_of_mind_sentences(token,
                    doc, theory_of_mind_sentences)
        return theory_of_mind_sentences

    def theory_of_mind_sentences(self, token, hdoc, theory_of_mind_sentences):
        if token._.vwp_cognitive or token._.vwp_communication \
           or token._.vwp_emotion or token._.vwp_emotional_impact:
            for child in token.children:
                if (child.dep_ in complements \
                    or child.dep_ in clausal_modifier_dependencies) \
                   and token._.has_governing_subject \
                   and not child._.vwp_quoted:
                    controllers = ResolveReference(
                        hdoc[token._.governing_subject], hdoc)
                    for controller in controllers:
                        agent = hdoc[controller]
                        if child._.has_governing_subject:
                            childSubj = hdoc[child._.governing_subject]
                            childControllers = ResolveReference(
                                hdoc[child._.governing_subject], hdoc)
                            if agent._.animate \
                               and childControllers is not None \
                               and controller not in childControllers \
                               and childSubj._.animate \
                               and not (agent.lower_
                                        in first_person_pronouns
                                        and childSubj.lower_
                                        in first_person_pronouns) \
                               and not (agent.lower_
                                        in second_person_pronouns
                                        and childSubj.lower_
                                        in second_person_pronouns):

                                entry = \
                                    newSpanEntry('social awareness',
                                                 token.sent.start,
                                                 token.sent.end-1,
                                                 hdoc,
                                                 'theory of mind sentence')
                                if entry not in theory_of_mind_sentences:
                                    theory_of_mind_sentences.append(
                                        entry)
                                break
                    break
        return theory_of_mind_sentences

    def negativePredicate(self, item: Token):
        """
         This function identifies lexical predicates that
         function as equivalent to negation when combined
         with other elements. This list may not be complete
         -- to double check later.
        """
        if item.lemma_ in ['lack',
                           'fail',
                           'failure',
                           'absence',
                           'shortage',
                           'false',
                           'wrong',
                           'inaccurate',
                           'incorrect']:
            return True
        return False

    def findClause(self, token: Token):
        if token.tag_.startswith('V') \
           or isRoot(token):
            return token
        return self.findClause(token.head)

    def getFirstChild(self, token: Token):
        for child in token.children:
            return child
        else:
            return None

    def propagateNegation(self, doc: Doc):
        """
         Set baseline sentiment on lexical elements.
         Identify negation elements in the parse tree (to use
         to reverse sentiment polarities where negated)
        """
        negation_tokens = []
        for tok in doc:
            tok._.vwp_sentiment_ = tok._.sentiword
            # we combine data from the sentiword and polarity ratings
            # to come up with a more reliable estimate of tone, positive
            # or negative. We trust those estimates a lot more for words
            # marked as subjective (vwp_evaluation, vwp_hege, or in assessments
            # from spacytextblob). In that case, we take the absolutely
            # larger or the sentiword or polarity rating. Otherwise, we
            # take the minimum, because sentiword in particular seems to
            # be biased toward positive ratings for words that are actually
            # neutral.
            if tok._.vwp_evaluation \
               or tok._.vwp_hedge \
               or tok.text in doc._.assessments:
                if tok._.polarity < 0 or tok._.sentiword < 0:
                    tok._.vwp_tone_ = min(tok._.polarity, tok._.sentiword)
                elif tok._.polarity > 0 and tok._.sentiword > 0:
                    tok._.vwp_tone_ = max(tok._.polarity, tok._.sentiword)
                else:
                    tok._.vwp_tone_ = (tok._.polarity + tok._.sentiword) / 2
            else:
                tok._.vwp_tone_ = min(tok._.polarity, tok._.sentiword)

            # rule order fixes to the tone variable are generally a bad idea,
            # but these are so common that fixing them gets rid of a lot of
            # distraction when displaying positive and negative tone words
            # able certain pretty kind mean fun
            if tok.lower_ in ['able', 'ready'] \
               and tok.i + 1 < len(doc) \
               and tok.nbor(1) is not None \
               and tok.nbor(1).lower_ == 'to':
                tok._.vwp_tone_ = 0.0
            elif (tok.lower_ == 'fun'
                  and tok.i + 1 < len(doc)
                  and tok.nbor(1) is not None
                  and tok.nbor(1).lower_ == 'of'):
                tok._.vwp_tone_ = -1*tok._.vwp_tone_
                tok.nbor(1)._.vwp_tone_ = tok._.vwp_tone_
            elif tok.lower_ == 'certain' and tok.i < tok.head.i:
                tok._.vwp_tone_ = 0.0
            elif tok.lower_ == 'pretty' and tok.pos_ == 'ADV':
                tok._.vwp_tone_ = 0.0
            elif tok.lower_ in ['kind', 'right'] and tok.pos_ == 'NOUN':
                tok._.vwp_tone_ = 0.0
            elif tok.lower_ == 'mean' and tok.pos_ == 'VERB':
                tok._.vwp_tone_ = 0.0

            penultimate = None
            antepenultimate = None
            if tok.dep_ == 'neg' \
               or self.negativePredicate(tok) \
               or (tok.dep_ == 'preconj'
                   and tok.lower_ == 'neither') \
               or tok.lower_ == 'hardly' \
               or tok.lower_ == 'no' \
               or (antepenultimate is not None
                   and antepenultimate.lower_ == 'less'
                   and penultimate is not None
                   and penultimate.lower_ == 'than'):
                newTk = self.findClause(tok)
                if newTk not in negation_tokens:
                    negation_tokens.append(newTk)
                else:
                    negation_tokens.remove(newTk)

            if (antepenultimate is not None
                and antepenultimate.lower_ == 'less'
                or antepenultimate is not None
                and antepenultimate.lower_ == 'more') \
               and penultimate is not None \
               and penultimate.lower_ == 'than':
                antepenultimate.norm_ = tok.norm_

            antepenultimate = penultimate
            penultimate = tok
        return negation_tokens

    def traverseTree(self, token: Token, negation_tokens: list):
        """
         Traverse tree and call function to reverse sentiment polarity
         when negated
        """

        if token.pos_ in ['ADJ', 'ADV'] \
           and token.lower_ in pos_degree_mod \
           and token._.vwp_tone_ is not None \
           and token.head._.vwp_tone_ is not None \
           and token._.vwp_tone_ > 0 \
           and token.head._.vwp_tone_ < 0:
            token._.vwp_tone_ = -1 * token._.vwp_tone_

        # negation takes scope over everything it governs, i.e., its
        # children and its sister nodes and their children
        # so if we have a negation, we need to apply it not just to
        # the node and its children, but to all children of the node
        # that is that child's head
        if token in negation_tokens:
            if token is not None:
                neg = self.getFirstChild(token)
                self.spread_reverse_polarity(token)
                negation_tokens.remove(token)

        # once we have handled negation at this scope level,
        # handle any embedded negations
        if len(negation_tokens) > 0:
            for child in token.children:
                if child != token:
                    negation_tokens = \
                        self.traverseTree(child, negation_tokens)

        return negation_tokens

    def spread_reverse_polarity(self, tok: Token):
        """
         This function traverses the children of a node and marks
         # them all as negated, except for most right children of
         # nouns, which need to be treated as not affected
         by negation.
        """

        # negate the node
        tok._.vwp_sentiment_ = -1 * tok._.vwp_sentiment_

        lastChild = None

        # now traverse the children of the node
        for child in tok.children:

            if child != tok and not tensed_clause(child):

                # and recurse over children of the children
                self.spread_reverse_polarity(child)
                lastChild = child

    def vwp_interactive(self, doc: Doc):
        interactiveStatus = []
        for token in doc:
            entry = newTokenEntry('interactive', token)
            entry['value'] = False
            entry2 = None

            # colloquial usage/slang
            if token._.usage:
                entry['value'] = True               

            if token.lower_ in first_person_pronouns:
                entry['value'] = True               

            elif token.lower_ in second_person_pronouns:
                entry['value'] = True               

            elif wh_question_word(token):
                entry['value'] = True               

            elif contraction(token):
                entry['value'] = True               

            # Preposition stranding
            elif (token.dep_ == 'IN'
                  and 'pobj' not in [child.dep_
                                     for child in token.children]
                  and 'prep' not in [child.dep_
                                     for child in token.children]):
                entry['value'] = True               

            # Anaphoric use of auxiliaries
            elif (token.pos == 'AUX'
                  and 'VERB' not in [child.pos_ for child
                                     in token.head.children]):
                entry['value'] = True               

            elif contracted_verb(token):
                entry['value'] = True               

            elif (token.lower_ in demonstratives
                  and token.dep_ != 'mark'
                  and token.tag_ != 'WDT'
                  and token.pos_ in ['PRON', 'DET']):
                entry['value'] = True               

            elif token.lower_ in indefinite_pronoun:
                entry['value'] = True               

            # Use of a conjunction to start a main clause
            elif token.is_sent_start and token.tag_ == 'CCONJ':
                entry['value'] = True               

            elif illocutionary_tag(token):
                entry['value'] = True               

            elif emphatic_adverb(token):
                entry['value'] = True               

            elif emphatic_adjective(token):
                entry['value'] = True               

            elif common_evaluation_adjective(token):
                entry['value'] = True               

            elif common_hedge_word(token):
                entry['value'] = True               

            elif elliptical_verb(token):
                entry['value'] = True               
                entry2 = newTokenEntry('interactive', token.nbor())
                entry2['value'] = True

            elif (token.lemma_ in ['lot', 'bit', 'while', 'ways']
                  and token.dep_ == 'npadvmod'):
                entry['value'] = True               


            elif token.pos_ == 'DET' and token.dep_ == 'advmod':
                # expressions like 'all in all'
                entry['value'] = True               

            elif token.dep_ == 'predet':
                # predeterminers like 'such a', 'what', or 'quite'
                entry['value'] = True               

            elif indefinite_comparison(token):
                entry['value'] = True               

            elif absolute_degree(token):
                entry['value'] = True               

            elif (token.lemma_ == 'old'
                  and self.getFirstChild(token.head) is not None
                  and 'any' == self.getFirstChild(token.head).lemma_):
                  # any old
                entry['value'] = True               

            elif (token.lemma_ in ['bunch', 'couple', 'lot']
                  and self.getFirstChild(token) is not None
                  and 'of' == self.getFirstChild(token).lemma_):
                entry['value'] = True               

            # Use of common private mental state verbs with
            # first or second person pronouns
            elif private_mental_state_tag(token):
                entry['value'] = True               
                entry2 = newTokenEntry('interactive', token.head)
                entry2['value'] = True

            # Use of words with a strong conversational flavor
            elif token.lemma_ in other_conversational_vocabulary:
                entry['value'] = True               

            # Use of interjections
            elif token.pos_ == 'INTJ':
                entry['value'] = True               

            elif other_conversational_idioms(token):
                entry['value'] = True               

                if token.i + 1 < len(doc):
                    entry2 = newTokenEntry('interactive', token)
                    entry2['value'] = True

            # Use of idiomatic prep + adj combinations like
            # for sure, for certain, for good
            elif token.pos_ == 'ADJ' and token.head.pos_ == 'ADP':
                entry['value'] = True               
            interactiveStatus.append(entry)
            if entry2 is not None:
                interactiveStatus.append(entry2)
        return interactiveStatus
        
    def nominalReferences(self, doc):
        """
        A listing of potential entities. No proper
        resolution of nominal reference. If the two different
        entities are referred to by the same noun, or two different
        nouns used for the same entity, we don't attempt to resolve
        them. TO-DO: a more proper reference resolver that handles
        nouns, instead of just the anaphora resolution handled by
        coreferee.
        """
       
        characterList = {}
        referenceList = {}
        registered = []
        for token in doc:
            if token._.animate or token.lower_ in characterList:
                if token.pos_ == 'PROPN' \
                   and token.ent_type_ not in nonhuman_ent_type:
                    if token.text.capitalize() in characterList:
                        if token.i not in characterList[
                           token.text.capitalize()]:
                            characterList[
                                token.text.capitalize()].append(token.i)
                            registered.append(token.i)
                    elif token.orth_.capitalize() not in characterList:
                        characterList[token.orth_.capitalize()] = [token.i]
                        registered.append(token.i)
                    else:
                        if token.i not in characterList[
                           token.orth_.capitalize()]:
                            characterList[
                                token.orth_.capitalize()].append(token.i)
                            registered.append(token.i)
                elif token.pos_ == 'NOUN':
                    # Not animate if a noun! Pronoun/noun confusion case
                    if token.lower_ == 'mine':
                        continue
                    if token._.root is not None \
                       and token._.root.capitalize() not in characterList:
                        characterList[token._.root.capitalize()] = [token.i]
                        registered.append(token.i)
                    elif token.text.capitalize() not in characterList:
                        characterList[token.text.capitalize()] = [token.i]
                        registered.append(token.i)
                    else:
                        if token.i not in characterList[
                           token.text.capitalize()]:
                            characterList[
                                token.text.capitalize()].append(token.i)
                            registered.append(token.i)
                elif token.pos_ == 'PRON':
                    antecedent = [doc[loc]
                                  for loc in ResolveReference(token, doc)]
                    if len(antecedent) > 0 and antecedent[0].i == token.i:
                        if not token._.vwp_quoted \
                           and token.lower_ in first_person_pronouns:
                            if 'SELF' not in characterList:
                                characterList['SELF'] = [token.i]
                                registered.append(token.i)
                            else:
                                if token.i not in characterList['SELF']:
                                    characterList['SELF'].append(token.i)
                                    registered.append(token.i)
                        elif (not token._.vwp_quoted
                              and token.lower_ in second_person_pronouns):
                            if 'You' not in characterList:
                                characterList['You'] = [token.i]
                                registered.append(token.i)
                            else:
                                if token.i not in characterList['You']:
                                    characterList['You'].append(token.i)
                                    registered.append(token.i)
                        else:
                            if not token._.vwp_quoted:
                                if token.lower_ in third_person_pronouns:
                                    antecedents = \
                                        scanForAnimatePotentialAntecedents(
                                            doc, token.i, [])
                                    charname = ''
                                    lems = []
                                    for i, ref in enumerate(antecedents):
                                        if doc[ref].text not in lems:
                                            lems.append(
                                                doc[ref].text.capitalize())
                                        if i == 0:
                                            charname = \
                                                doc[ref].text.capitalize()
                                        elif i+1 == len(antecedents):
                                            charname += ' and ' \
                                                + doc[ref].text.capitalize()
                                        else:
                                            charname += ', ' \
                                                + doc[ref].text.capitalize()
                                    if charname not in characterList:
                                        characterList[charname] = []
                                    characterList[charname].append(token.i)
                                    registered.append(token.i)
                    elif (token.lower_ in third_person_pronouns
                          and len(antecedent) > 1):
                        charname = ''
                        lems = []
                        for i, ref in enumerate(antecedent):
                            if ref.text not in lems:
                                lems.append(ref.text.capitalize())
                                if i == 0:
                                    charname = ref.text.capitalize()
                                elif i+1 == len(antecedent):
                                    charname += ' and ' + ref.text.capitalize()
                                else:
                                    charname += ', ' + ref.text.capitalize()
                        if charname not in characterList:
                            characterList[charname] = []
                        characterList[charname].append(token.i)
                        registered.append(token.i)

                    elif (len(antecedent) > 0
                          and antecedent[0].lemma_.capitalize
                          in characterList):
                        if token.i not in characterList[antecedent[0].lemma_]:
                            characterList[
                                antecedent[0].lemma_.capitalize()
                                ].append(token.i)
                            registered.append(token.i)
                    elif (len(antecedent) > 0
                          and antecedent[0].text.capitalize()
                          in characterList):
                        if token.i not in characterList[
                           antecedent[0].text.capitalize()
                           ]:
                            characterList[
                                antecedent[0].text.capitalize()
                                ].append(token.i)
                            registered.append(token.i)
                    elif len(antecedent) > 0 \
                        and antecedent[0].orth_.capitalize() \
                            in characterList:
                        if token.i not in characterList[
                           antecedent[0].orth_.capitalize()]:
                            characterList[
                                antecedent[0].orth_.capitalize()
                                ].append(token.i)
                            registered.append(token.i)
                    elif len(antecedent) > 0:
                        characterList[
                            antecedent[0].lemma_.capitalize()
                            ] = [token.i]
                        registered.append(token.i)
            else:
                if token.pos_ == 'PROPN' \
                    and token.ent_type_ not in ['ORG',
                                                'DATE']:
                    if token.text.capitalize() in characterList:
                        if token.i not in characterList[
                           token.text.capitalize()]:
                            characterList[
                                token.text.capitalize()].append(token.i)
                            registered.append(token.i)
                    elif token.orth_.capitalize() not in characterList:
                        characterList[token.orth_.capitalize()] = [token.i]
                        registered.append(token.i)
                    else:
                        if token.i not in characterList[
                           token.orth_.capitalize()]:
                            characterList[
                                token.orth_.capitalize()].append(token.i)
                            registered.append(token.i)
                elif token.pos_ == 'NOUN':
                    if token._.root is not None \
                       and token._.root.capitalize() not in referenceList:
                        referenceList[token._.root.capitalize()] = [token.i]
                        registered.append(token.i)
                    elif token.text.capitalize() not in referenceList:
                        referenceList[token.text.capitalize()] = [token.i]
                        registered.append(token.i)
                    else:
                        if token.i not in referenceList[
                           token.text.capitalize()]:
                            referenceList[
                                token.text.capitalize()].append(token.i)
                            registered.append(token.i)
                elif token.pos_ == 'PRON':
                    antecedent = [doc[loc] for loc
                                  in ResolveReference(token, doc)]
                    if antecedent[0].i == token.i:
                        if not token._.vwp_quoted \
                           and token.lemma_ in first_person_pronouns:
                            if 'SELF' not in characterList:
                                characterList['SELF'] = [token.i]
                                registered.append(token.i)
                            else:
                                if token.i not in characterList['SELF']:
                                    characterList['SELF'].append(token.i)
                                    registered.append(token.i)
                        elif not token._.vwp_quoted \
                            and token.lemma_ in second_person_pronouns:
                            if 'You' not in characterList:
                                characterList['You'] = [token.i]
                                registered.append(token.i)
                            else:
                                if token.i not in characterList['You']:
                                    characterList['You'].append(token.i)
                                    registered.append(
                                        token.i)
                        else:
                            if not token._.vwp_quoted:
                                if token.lower_ in third_person_pronouns:
                                    antecedents = \
                                        scanForAnimatePotentialAntecedents(
                                            doc, token.i, [])
                                    charname = ''
                                    lems = []
                                    for i, ref in enumerate(antecedents):
                                        if doc[ref].text not in lems:
                                            lems.append(
                                                doc[ref].text.capitalize())
                                        if i == 0:
                                            charname = \
                                                doc[ref].text.capitalize()
                                        elif i+1 == len(antecedents):
                                            charname += ' and ' \
                                                + doc[ref].text.capitalize()
                                        else:
                                            charname += ', ' \
                                                + doc[ref].text.capitalize()
                                    if charname not in characterList:
                                        characterList[charname] = []
                                    characterList[charname].append(token.i)
                                    registered.append(token.i)
                    elif (token.lower_ in third_person_pronouns
                          and len(antecedent) > 1):
                        charname = ''
                        lems = []
                        for i, ref in enumerate(antecedent):
                            if ref.text not in lems:
                                lems.append(ref.text.capitalize())
                                if i == 0:
                                    charname = ref.text.capitalize()
                                elif i+1 == len(antecedent):
                                    charname += ' and ' + ref.text.capitalize()
                                else:
                                    charname += ', ' + ref.text.capitalize()
                        if charname not in referenceList:
                            referenceList[charname] = []
                        referenceList[charname].append(token.i)
                        registered.append(token.i)
                    elif antecedent[0].i not in registered:
                        if antecedent[0].lemma_.capitalize() in referenceList:
                            if token.i not in referenceList[
                               antecedent[0].lemma_.capitalize()]:
                                referenceList[
                                              antecedent[0].lemma_.capitalize()
                                              ].append(token.i)
                                registered.append(token.i)
                        elif antecedent[0].text.capitalize() in referenceList:
                            if token.i not in referenceList[
                               antecedent[0].text.capitalize()]:
                                referenceList[
                                              antecedent[0].text.capitalize()
                                              ].append(token.i)
                                registered.append(token.i)
                        elif antecedent[0].orth_.capitalize() in referenceList:
                            if token.i not in referenceList[
                               antecedent[0].orth_.capitalize()]:
                                referenceList[
                                              antecedent[0].orth_.capitalize()
                                              ].append(token.i)
                                registered.append(token.i)
                        elif len(antecedent) > 0:
                            referenceList[
                                antecedent[0].lemma_.capitalize()] = [token.i]
                            registered.append(token.i)
        directspeech = self.vwp_direct_speech(doc)
        for character in characterList:
            for speechevent in directspeech:
                speaker = speechevent['value'][0]
                addressee = speechevent['value'][1]
                for item in speaker:
                    if item in characterList[character]:
                        for item2 in speaker:
                            if item2 not in characterList[character]:
                                characterList[character].append(item2)
                for item in addressee:
                    if item in characterList[character]:
                        for item2 in addressee:
                            if item2 not in characterList[character]:
                                characterList[character].append(item2)
        return (characterList, referenceList)

    def tense_changes(self, document):
        tenseChanges = []
        currentEvent = {}
        i = 0
        past_tense_state = False
        
        # We need to call this attribute to make
        # sure that set_direct_speech spans has been
        # called before we check tense sequences
        self.vwp_direct_speech(document)
        
        while i < len(document):
            if i > 0 \
               and not in_past_tense_scope(
                   getTensedVerbHead(document[i])) \
               and not document[i].tag_ in loose_clausal_dependencies \
               and not document[i-1].tag_ in loose_clausal_dependencies \
               and in_past_tense_scope(getTensedVerbHead(document[i-1])):
                if not document[i]._.vwp_in_direct_speech_:
                    if past_tense_state:
                        currentEvent['loc'] = i
                        currentEvent['past'] = False
                        tenseChanges.append(currentEvent)
                        currentEvent = {}
                        past_tense_state = False
            elif (i > 1
                  and not in_past_tense_scope(
                      getTensedVerbHead(document[i]))
                  and not document[i].tag_ in loose_clausal_dependencies
                  and document[i-1].tag_ in loose_clausal_dependencies
                  and not document[i-2].tag_ in loose_clausal_dependencies
                  and in_past_tense_scope(getTensedVerbHead(document[i-2]))):
                if not document[i]._.vwp_in_direct_speech_:
                    if past_tense_state:
                        currentEvent['loc'] = i
                        currentEvent['past'] = False
                        tenseChanges.append(currentEvent)
                        currentEvent = {}
                        past_tense_state = False

            elif (i == 0
                  and in_past_tense_scope(getTensedVerbHead(document[i]))):
                if not document[i]._.vwp_in_direct_speech_:
                    past_tense_state = True
                    currentEvent['loc'] = i
                    currentEvent['past'] = True
                    tenseChanges.append(currentEvent)
                    currentEvent = {}

            elif (i > 0
                  and not document[i-1].tag_ in loose_clausal_dependencies
                  and not in_past_tense_scope(
                      getTensedVerbHead(document[i-1]))
                  and not document[i].tag_ in loose_clausal_dependencies
                  and in_past_tense_scope(
                      getTensedVerbHead(document[i]))):
                if not document[i]._.vwp_in_direct_speech_:
                    if not past_tense_state:
                        currentEvent['loc'] = i
                        currentEvent['past'] = True
                        tenseChanges.append(currentEvent)
                        currentEvent = {}
                        past_tense_state = True
            elif (i > 1
                  and not document[i-2].tag_ in loose_clausal_dependencies
                  and not in_past_tense_scope(
                      getTensedVerbHead(document[i-2]))
                  and document[i-1].tag_ in loose_clausal_dependencies
                  and not document[i].tag_ in loose_clausal_dependencies
                  and in_past_tense_scope(
                      getTensedVerbHead(document[i]))):
                if not document[i]._.vwp_in_direct_speech_:
                    if not past_tense_state:
                        currentEvent['loc'] = i
                        currentEvent['past'] = True
                        past_tense_state = True
                        tenseChanges.append(currentEvent)
                        currentEvent = {}
            i += 1
        return tenseChanges

    def concrete_details(self, doc):
        characterList = None
        (characterList, referenceList) = self.nominalReferences(doc)
        detailList = []

        # We need to call this attribute to make
        # sure that set_direct_speech spans has been
        # called before we check for concrete details
        self.vwp_direct_speech(doc)

        for token in doc:
            entry = newTokenEntry('concrete_detail', token)
            entry['value'] = False

            if token._.vwp_abstract \
               or token._.is_academic \
               or token._.is_latinate:
                detailList.append(entry)
                continue

            # Higher frequency words aren't likely to be concrete details
            if token._.max_freq > 5:
                detailList.append(entry)
                continue

            if token._.is_academic:
                detailList.append(entry)
                continue

            if token.text.capitalize() in characterList \
               and len(characterList[token.text.capitalize()]) > 2:
                detailList.append(entry)
                continue
            if token._.vwp_direct_speech_verb_ \
               or token._.vwp_in_direct_speech_:
                detailList.append(entry)
                continue
            if token._.has_governing_subject \
               and doc[token._.governing_subject]._.animate \
               and (generalViewpointPredicate(token)
                    or token._.vwp_emotion):
                detailList.append(entry)
                continue
            if getLogicalObject(token) is not None \
               and getLogicalObject(token)._.animate \
               and token._.vwp_emotional_impact:
                detailList.append(entry)
                continue
            if (token._.vwp_evaluation
                or token._.vwp_hedge
                or (token._.abstract_trait
                    and 'of' in [child.lower_
                                 for child in token.children])):
                detailList.append(entry)
                continue
            if (token.pos_ != 'PROPN'
                and (token.dep_ != 'punct'
                     and not token.is_stop)):
                if (token.pos_ == 'NOUN') and \
                   (token._.max_freq is None
                   or token._.nSenses is None
                   or token._.max_freq < 4.5) \
                   and token._.concreteness is not None \
                   and token._.concreteness >= 4:
                    entry['value'] = True
                    detailList.append(entry)
                elif ((token.pos_ == 'VERB') and
                      token._.max_freq is not None
                      and token._.max_freq < 4.2
                      and token._.concreteness is not None
                      and (token._.concreteness >= 3.5
                           or token._.concreteness >= 2.5
                           and getLogicalObject(token) is not None
                           and getLogicalObject(token)._.concreteness
                           is not None
                           and getLogicalObject(token)._.concreteness
                           >= 4.3)):
                    entry['value'] = True
                    detailList.append(entry)
                elif token.pos_ != 'VERB' and token.pos_ != 'NOUN':
                    if token._.has_governing_subject \
                       and doc[
                               token._.governing_subject
                               ]._.concreteness is not None \
                       and doc[token._.governing_subject]._.concreteness >= 4 \
                       and token._.concreteness is not None \
                       and token._.concreteness > 2.5 \
                       and (token._.max_freq is None
                            or token._.max_freq < 4.3
                            or token._.nSenses is None
                            or token._.nSenses < 4
                            and token._.max_freq < 5):
                        entry['value'] = True
                        detailList.append(entry)
                    elif (token.head._.concreteness is not None
                          and token.head._.concreteness >= 4
                          and token._.concreteness is not None
                          and token._.concreteness > 2.5
                          and (token._.max_freq is None
                               or token._.max_freq < 4.3
                               or token._.nSenses is None
                               or token._.nSenses < 4
                               and token._.max_freq < 5)):
                        entry['value'] = True
                        detailList.append(entry)
        return detailList
