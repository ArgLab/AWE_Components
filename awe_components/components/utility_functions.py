#!/usr/bin/env python3
# Copyright 2022, Educational Testing Service

import sys
import logging
import spacy
import statistics
import numpy as np
import re
import math
import pandas as pd
import json

from enum import Enum
from spacy.tokens import Token, Doc, Span
from nltk.corpus import wordnet as wn
from ..errors import *
from awe_components.wordprobs.wordseqProbClient import *

logging.basicConfig(level="DEBUG")


class FType(Enum):
    """
     Types of summary features we can create for most of our metrics
    """
    STDEV = 1
    MEAN = 2
    MEDIAN = 3
    MAX = 4
    MIN = 5

built_in_attributes = ['text',
                       'text_with_ws',
                       'orth_',
                       'ent_type_',
                       'ent_iob_',
                       'ent_id',
                       'lemma_',
                       'norm_',
                       'lower_',
                       'is_alpha',
                       'is_ascii',
                       'is_digit',
                       'is_lower',
                       'is_upper',
                       'is_title',
                       'is_punct',
                       'is_left_punct',
                       'is_right_punct',
                       'is_sent_start',
                       'is_sent_end',
                       'is_space',
                       'is_bracket',
                       'is_quote',
                       'is_currency',
                       'like_url',
                       'like_num',
                       'like_email',
                       'is_oov',
                       'is_stop',
                       'pos_',
                       'tag_',
                       'dep_',
                       'lang_',
                       'idx']

built_in_string_functions = ['text',
                       'text_with_ws',
                       'orth_',
                       'ent_type_',
                       'lemma_',
                       'norm_',
                       'lower_',
                       'pos_',
                       'tag_',
                       'dep_',
                       'lang_']

built_in_flags = ['is_alpha',
                  'is_ascii',
                  'is_digit',
                  'is_lower',
                  'is_upper',
                  'is_title',
                  'is_punct',
                  'is_left_punct',
                  'is_right_punct',
                  'is_sent_start',
                  'is_sent_end',
                  'is_space',
                  'is_bracket',
                  'is_quote',
                  'is_currency',
                  'like_url',
                  'like_num',
                  'like_email',
                  'is_oov',
                  'is_stop']

docspan_extensions = ['sentence_types', 
                      'transition_distances',
                      'intersentence_cohesions',
                      'sliding_window_cohesions',
                      'corefChainInfo',
                      'sentenceThemes',
                      'transitions',
                      'syntacticDepthsOfThemes',
                      'syntacticDepthsOfRhemes',
                      'main_cluster_spans',
                      'vwp_statements_of_opinion',
                      'vwp_statements_of_fact',
                      'direct_speech_spans',
                      'vwp_social_awareness',
                      'vwp_propositional_attitudes',
                      'main_ideas',
                      'supporting_ideas',
                      'supporting_details',
                      'all_cluster_info'
                      ]
        
def lexFeat(tokens, theProperty):
    """
    Return feature values based on sets of attribute values
    coded as extensions to the spacy token object
    """
    theSet = [float(t._.get(theProperty)) for t in tokens
              if t._.get(theProperty) is not None
              and not t.is_stop
              and t.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV']]
    return theSet


def summarize(items, summaryType=FType.STDEV):
    """
     General service function to calculate summary features from raw data
    """
    filtered = []
    for item in items:
        if item is not None:
            filtered.append(item)
    if len(filtered) == 0:
        return None
    if summaryType == FType.MEAN:
        return statistics.mean(filtered)
    elif summaryType == FType.MEDIAN:
        return statistics.median(filtered)
    elif summaryType == FType.STDEV:
        if len(items) > 2:
            return statistics.stdev(filtered)
        else:
            return None
    if summaryType == FType.MAX:
        return max(filtered)
    if summaryType == FType.MIN:
        return min(filtered)


def print_parse_tree(sent):
    """
        Print pretty formatted version of parse tree
    """

    lastToken = None
    headLoc = 0
    lastHeadLoc = 0

    ########################################################################
    # print each token in the sentence in turn with appropriate annotation #
    ########################################################################
    for token in sent:

        # set up some useful variables
        head = getHead(token)
        depth = getDepth(token)
        rightNeighbor = getRight(sent, token.i)

        # the actual depth we want to indent, as opposed to depth
        # in the parse tree
        usedDepth = getAdjustedDepth(token)

        ##############################################################
        # output the basic category status of the phrasal elements   #
        ##############################################################
        cat = ''

        # special case -- punctuation
        if token.dep_ == 'advmod' \
           and rightNeighbor is not None \
           and rightNeighbor.dep_ == 'punct':
            if token.tag_.lower().startswith('R'):
                cat = 'RB'
            else:
                cat = 'AP'

        # special case -- gerunds
        elif (token.dep_ == 'xcomp'
              and token.tag_ == 'vbg'):
            cat = 'SG'

        # special case -- auxiliaries at depth zero in the parse tree
        elif (depth == 0
              and (token.tag_ == 'BEZ'
                   or token.tag_ == 'BEM'
                   or token.tag_ == 'BER'
                   or token.tag_.startswith('HV')
                   or token.tag_.startswith('DO'))):
            cat = 'VP'

        # main branch of logic. The firstLeftsister function
        # helps us find the leftmost member of a np, vp, or
        # pp etc. span
        elif firstLeftSister(token, lastToken):
            if token.tag_ == 'VBG' and \
               token.dep_ == 'csubj':
                cat = 'SG'  # gerund
            elif token.tag_ == 'WRB':  # wh adverbs
                if head is not None:
                    for child in head.children:
                        if child.tag_ == 'TO':
                            cat = 'SI RB'
                            # infinitival clause with wh adverb
                        else:
                            cat = 'SB'
                            # subordinate clause with wh adverb
                else:
                    cat = 'SB'
                    # subordinate clause with wh adverb
            elif (token.tag_ == 'TO'
                  and head is not None
                  and head.dep_ == 'xcomp'):
                cat = 'SI'  # infinitive clause
            elif (token.dep_ == 'mark'
                  and head is not None
                  and (head.dep_ == 'advcl')):
                cat = 'SB\tCOMP'  # adverbial subordinate clause
            elif (token.dep_ == 'mark'
                  and head is not None
                  and (head.dep_ == 'ccomp'
                       or head.dep_ == 'acl'
                       or head.dep_ == 'csubj')):
                cat = 'SC\tCOMP'   # complement clause
            elif (token.dep_ == 'mark'
                  and head is not None
                  and head.dep_ == 'relcl'):
                cat = 'SR\tCOMP'   # relative clause with that
            elif token.tag_ == 'WDT':
                cat = 'SR\tNP'  # relative clause with wh determiner
            elif token.tag_ == 'WPS':
                cat = 'SR\tNP'  # relative clause with wh pronoun
            elif (token.tag_.startswith('V')
                  and token.dep_ == 'conj'
                  and head is not None
                  and head.dep_ == 'advcl'):
                cat = 'SB'
                # adverbial subordinate clause
                # in compound structure
            elif (token.tag_.startswith(' ')
                  and token.dep_ == 'conj'
                  and head is not None
                  and head.dep_ == 'ccomp'):
                cat = 'SC'   # complement clause in compound structure
            elif (token.tag_.startswith('V')
                  and token.dep_ == 'conj'
                  and head is not None
                  and head.dep_ == 'acl'):
                cat = 'SC'  # compound clause in compound structure
            elif (token.tag_.startswith('V')
                  and token.dep_ == 'conj'
                  and head is not None
                  and head.dep_ == 'relcl'):
                cat = 'SR'  # relative clause
            elif (token.tag_.startswith('V')
                  and token.dep_ == 'conj'
                  and head is not None
                  and head.dep_ == 'xcomp'):
                cat = 'SJ'  # conjoined main clause or VP
            elif (token.tag_ == 'CC'
                  and head is not None
                  and isRoot(head)):
                cat = 'SJ'  # conjoined main clause or VP
            elif token.tag_ == 'CC':
                cat = 'CC'  # coordinating conjunction
            elif (token.dep_ == 'prep'
                  or token.dep_ == 'agent'):
                cat = 'PP'  # prepositional phrase
            elif (token.dep_ == 'acomp'
                  or (token.dep_ == 'neg'
                      and head is not None
                      and (head.tag_.startswith('J')
                           or head.tag_.startswith('R')))
                  or (token.dep_ == 'advmod'
                      and (head is not None
                           and head.dep_ != 'amod'
                           and (head.i < token.i
                                or head.tag_.startswith('R')
                                or head.tag_.startswith('J'))))):
                if (token.tag_.lower().startswith('R')):
                    cat = 'RB'  # adverb or adverb phrase
                else:
                    cat = 'AP'  # adjective phrase
            elif (token.dep_ == 'det'
                  or (token.dep_ == 'neg'
                      and head is not None
                      and head.dep_ == 'det')
                  or token.dep_ == 'poss'
                  or token.dep_ == 'amod'
                  or token.dep_ == 'nummod'
                  or token.dep_ == 'compound'
                  or token.dep_ == 'nsubj'
                  or token.dep_ == 'nsubjpass'
                  or token.dep_ == 'dobj'
                  or token.dep_ == 'pobj'
                  or token.dep_ == 'appos'
                  or token.dep_ == 'attr'
                  or token.tag_.startswith('N')
                  or token.tag_.startswith('TUNIT')):
                cat = 'NP'  # noun phrase
            elif ((depth == 0
                   and not hasLeftChildren(token))
                  or token.dep_ == 'aux'
                  or token.dep_ == 'auxpass'
                  or token.dep_ == 'neg'
                  or (token.dep_ == 'advmod'
                      and token.i < head.i)
                  or token.dep == 'acl'
                  or token.dep_ == 'relcl'
                  or token.dep_ == 'advcl'
                  or token.dep_ == 'ccomp'
                  or token.tag_.startswith('V')
                  or token.tag_.startswith('BE')
                  or token.tag_.startswith('DO')
                  or token.tag_.startswith('HV')):
                cat = 'VP'  # verb phrase

        headLoc -= 1
        header = '\t'

        ################################################################
        # Set up the header element that captures category information #
        # and depth                                                    #
        ################################################################

        # mark the start of the sentence
        if isLeftEdge(token, sent):
            header += 'S'

        # add tabs to capture the degree of indent we are setting for
        # this word
        while usedDepth >= 0:
            header += '\t'
            usedDepth -= 1
        # put the category of the word as the first item in the indent
        header += cat

        headLoc = -1
        if head is not None:
            headLoc = head.i

        ##################################################################
        # format the whole line and print it. Index of word plus header  #
        # information including word category, followed by the token's   #
        # tag and text, its lemma in parentheses, followed by the de-    #
        # pendency label and the index of the word the dependency points #
        # to                                                             #
        ##################################################################
        anteced = ResolveReference(token, sent)
        line = str(token.i) \
            + header \
            + "\t|" \
            + token.tag_ \
            + " " \
            + token.text + \
            " (" \
            + token.lemma_.replace('\n', 'para') \
            + " " \
            + str(anteced) \
            + ")" \
            + " " \
            + token.dep_ + \
            ":" \
            + str(headLoc)

        line = line.expandtabs(6)
        print(line,
              'ant:', token._.antecedents,
              'gsubj:', token._.governing_subject,
              'vp:', token._.vwp_perspective)

        lastToken = token
        if head is not None:
            lastHeadLoc = head.i


def getHead(tok: Token):
    if tok is not None and tok is not bool:
        for anc in tok.ancestors:
            return anc
    return None


def getDepth(tok: Token):
    """
     This function calculates the depth of the current word
     in the spaCY dependency tree
    """
    depth = 0
    if tok is not None:
        for anc in tok.ancestors:
            depth += 1
    return depth


def getAdjustedDepth(tok: Token):
    """
     This function adjusts the depth of the word node to the
     depth we want to display in the output
    """
    depth = getDepth(tok)
    adjustment = 0
    if tok is not None:
        for anc in tok.ancestors:
            # clausal subjects need to be embedded one deeper
            # than other elements left of the head, but
            # otherwise we decrease indent of elements left of
            # the head to the indent of the head, to display
            # them as a single span
            if tok.i < anc.i \
               and anc.dep_ != 'csubj' \
               and tok.dep_ != 'csubj':
                adjustment += 1
            # clauses should be indented one level deeper
            # than the dependency tree suggests
            if tok.dep_ == 'advcl' \
               or tok.dep_ == 'ccomp' \
               or tok.dep_ == 'acl' \
               or tok.dep_ == 'relcl':
                adjustment -= 1
            if anc.dep_ == 'advcl' \
               or anc.dep_ == 'ccomp'\
               or anc.dep_ == 'acl' \
               or anc.dep_ == 'relcl':
                adjustment -= 1
    head = getHead(tok)
    if tok.dep_ == 'mark' \
       and head is not None \
       and head.dep_ == 'csubj':
        adjustment += 1
    return depth-adjustment


def getRight(sentence, loc):
    """
     This function returns the word immediately to the
     right of the input token
    """
    if loc + 1 < sentence.__len__():
        return sentence[loc + 1]
    return None


def firstLeftSister(tokenA: Token, tokenB: Token):
    """
     This function indicates that a word is the leftmost
     dependent of a head word. It requires a speries of
     special checks based upon knowledge that for instance
     'case' (possessive) and punctuations don't interrupt
     a phrase, and that each type of phrase uses a limited
     number of dependencies for left sisters
    """
    depth = getDepth(tokenA)
    depthB = getDepth(tokenB)
    head = getHead(tokenA)
    if abs(depth - depthB) > 1 \
       and (tokenB is None
            or tokenB.dep_ != 'case'
            and tokenB.dep_ != 'punct'):
        return True
    if tokenA is not None \
       and tokenB is None:
        return True
    elif (tokenA is not None
          and tokenB is not None):
        if tokenA.dep_ == 'prep' \
           and tokenB.tag_.startswith('R') \
           and tokenB.text.lower().endswith('ly'):
            return True
        if tokenA.dep_ == 'advmod' \
           and tokenA.tag_.startswith('R') \
           and head.tag_.startswith('V') \
           and head.i == tokenA.i - 1:
            return True
        if (tokenA.dep_ == 'aux'
            or tokenA.dep_ == 'auxpass'
            or tokenA.dep_ == 'neg'
            or tokenA.dep_ == 'advmod'
            or tokenA.dep_ == 'advcl'
            or tokenA.dep_ == 'relcl'
            or tokenA.dep_ == 'conj'
            or tokenA.tag_.startswith('V')
            or tokenA.tag_.startswith('BE')
            or tokenA.tag_.startswith('DO')
            or tokenA.tag_.startswith('HV')) \
           and (tokenB.dep_ == 'aux'
                or tokenB.dep_ == 'auxpass'
                or tokenB.dep_ == 'neg'
                or tokenB.dep_ == 'advmod'
                or (tokenB.dep_ == 'punct'
                    and tokenA in tokenB.ancestors)
                or tokenB.tag_.startswith('BE')
                or tokenB.tag_.startswith('DO')
                or tokenB.tag_.startswith('HV')
                or tokenB.tag_.startswith('V')):
            return False
        if (tokenA.dep_ == 'det'
            or tokenA.dep_ == 'poss'
            or tokenA.dep_ == 'amod'
            or tokenA.dep_ == 'nummod'
            or tokenA.dep_ == 'compound'
            or tokenA.dep_ == 'nsubj'
            or tokenA.dep_ == 'nsubjpass'
            or tokenA.dep_ == 'csubj'
            or tokenA.dep_ == 'csubjpass'
            or tokenA.dep_ == 'dobj'
            or tokenA.dep_ == 'pobj'
            or tokenA.dep_ == 'attr'
            or tokenA.dep_ == 'appos'
            or (tokenA.dep_ == 'neg'
                and getHead(tokenA) is not None
                and getHead(tokenA).dep_ == 'det')) \
           and (tokenB.dep_ == 'det'
                or tokenB.dep_ == 'poss'
                or tokenB.dep_ == 'amod'
                or tokenB.dep_ == 'nummod'
                or tokenB.dep_ == 'compound'
                or tokenB.dep_ == 'case'
                or (tokenB.dep_ == 'punct'
                    and tokenA in tokenB.ancestors)
                or (tokenB.dep_ == 'neg'
                    and getHead(tokenA) is not None
                    and getHead(tokenB).dep_ == 'det')):
            return False
        if (tokenA.dep_ == 'advmod'
            or tokenA.dep_ == 'acomp'
            or tokenA.dep_ == 'prep'
            or (tokenA.dep_ == 'neg'
                and head is not None
                and (head.tag_.startswith('J')
                     or head.tag_.startswith('R')))) \
           and (tokenB.dep_ == 'advmod'):
            return False
    return True


def isLeftEdge(token: Token, sentence: Span):
    """
     This function indicates whether a token is the
     leftmost element in a sentence
    """
    for tok in sentence:
        if tok == token:
            return True
        else:
            break
    return False


def hasLeftChildren(tok: Token):
    """
     This function indicates whether the token input to the
     function has any children to its left
    """
    for child in tok.children:
        if child.i < tok.i:
            return True
    return False


def leftSisterSpan(doc, start, end):
    """
     This function indicates that a sequence is the complete
     span from the first left dependent to the head
    """
    if end > start:
        return False
    lastToken = None
    while start < end:
        if lastToken is not None:
            if not firstLeftSister(doc[start], doc[start + 1]):
                return False
        start += 1
    return True


def getFirstChild(token: Token):
    for child in token.children:
        return child
    else:
        return None


def takesBareInfinitive(item: Token):
    """
     This function exists because spaCY uses the same dependency
     configuration for tensed clauses and untensed clauses (so-called
     "small clauses"). We need to know when something is a small clause
     so we know how to indent the tree properly, among other things.
     The list in this function may not be complete -- the correct
     list should be reviewed.
    """
    if item is None:
        return False
    if item.lemma_ in ["make",
                       "have",
                       "help",
                       "let",
                       "go",
                       "bid",
                       "feel",
                       "hear",
                       "see",
                       "watch",
                       "notice",
                       "observe",
                       "overhear",
                       "monitor",
                       "help",
                       "observe",
                       "perceive",
                       "notice",
                       "consider",
                       "proclaim",
                       "declare"]:
        return True
    return False


def tensed_clause(tok: Token):
    """
     This function calculates whether a token is the head of a tensed clause.
     We need to know if a clause is a tensed clause to count subordinate
     clauses and complement clauses correctly and indent them properly.
     Basically, tensed clauses are either built around the main verb of the
     sentence or they have a subject and are not infinitives or the complements
     of the verbs that take a bare infinitive, e.g., make, have, bid, and let
"""
    # does it have a subject?
    hasSubj = False
    hasTenseMarker = False
    infinitive = False
    head = getHead(tok)
    for child in tok.children:
        # tensed clauses obligatorily contain subjects
        if child.dep_ in ['nsubj', 'nsubjpass', 'csubj', 'csubjpass', 'expl']:
            hasSubj = True
        # infinitives are never tensed clauses
        if child.dep_ == 'aux' and child.tag_ == 'TO':
            hasTenseMarker = False
            infinitive = True
            break
        # a tensed clause has to contain a tensed verb,
        # which may be an auxiliary
        if child.dep_ == 'aux' \
           and (child.tag_ == 'MD'
                or child.lemma_ in ['am',
                                    'are',
                                    'is',
                                    'was',
                                    'were',
                                    'have',
                                    'has',
                                    'do',
                                    'does']
                or child.tag_ == 'BEZ'
                or child.tag_ == 'BEM'
                or child.tag_ == 'BER'
                or child.tag_.startswith('HV')
                or child.tag_.startswith('DO')
                or 'Tense=' in str(child.morph)):
            hasTenseMarker = True

    # if we're at root level, we still have to check if we have
    # a tensed verb, which may be an auxiliary
    if tok.tag_ == 'MD' \
       or tok.tag_ == 'BEZ' \
       or tok.tag_ == 'BEM' \
       or tok.tag_ == 'BER' \
       or tok.tag_.startswith('HV') \
       or tok.tag_.startswith('DO') \
       or 'Tense=' in str(tok.morph):
        hasTenseMarker = True

    if infinitive:
        return False
    # Imperatives count as tensed
    if not hasTenseMarker \
       and not hasSubj \
       and isRoot(tok) \
       and tok.text != tok.lemma_:
        return True
    # Otherwise subjectless verbs count as not tensed
    elif not hasSubj:
        return False
    # Otherwise inflected verbs count as not tensed
    elif (not hasTenseMarker
          and tok.tag_ != 'VBZ'
          and tok.text.lower() != tok.lemma_):
        return False
    # Subjects of small clauses (object + bare infinitive)
    # do not count as tensed
    elif (head is not None
          and takesBareInfinitive(head)
          and tok.text.lower() == tok.lemma_
          and hasSubj):
        return False
    return True


def tough_complement(node):
    if node.dep_ in ['xcomp', 'advcl']:
        if node._.vwp_tough:
            return True
        for child in node.head.children:
            if child.dep_ in ['acomp', 'attr'] \
               and child._.vwp_tough:
                return True
    return False


def raising_complement(node):
    if node.dep_ in ['xcomp', 'advcl']:
        if node._.vwp_raising:
            return True
        for child in node.head.children:
            if child.dep_ in ['acomp', 'attr'] \
               and child._.vwp_raising:
                return True
    return False


def past_tense_verb(tok: Token):
    """
    This function checks the space morphology feature
    to determine if a verb is in the past tense.
    """
    if 'Tense=Past' in str(tok.morph):
        for child in tok.children:
            if child.lemma_ == 'be':
                if child.text.lower() not in ['was', 'were']:
                    return False
        return True
    else:
        return False


def in_past_tense_scope(tok: Token):
    if tok is None:
        return None
    if '\n' in tok.text:
        return None
    if tok.text.lower() in ['was', 'were']:
        return True
    for item in tok.head.subtree:
        if item.dep_ == 'aux' \
           and item.text.lower() in ['do',
                                     'does',
                                     'has',
                                     'will',
                                     'can',
                                     'shall',
                                     'may',
                                     'must',
                                     'am',
                                     'are',
                                     'is']:
            return False
    if past_tense_verb(tok):
        return True
    while not isRoot(tok):
        if past_tense_verb(tok.head):
            return True
        elif tensed_clause(tok.head):
            for item in tok.head.subtree:
                if item.dep_ == 'aux' \
                   and item.text.lower() in ['did',
                                             'had',
                                             'was',
                                             'were',
                                             'could',
                                             'would',
                                             '\'d']:
                    return True
            return False
        if tok.head is not None and tok != tok.head:
            tok = tok.head
        else:
            return False
    for item in tok.head.subtree:
        if item.dep_ == 'aux' \
           and item.text.lower() in ['did',
                                     'had',
                                     '\'d',
                                     'was',
                                     'were',
                                     'could',
                                     'would',
                                     '’d']:
            return True
    if isRoot(tok) and 'VerbForm=Inf' in str(tok.morph):
        return False
    return False


def in_modal_scope(tok: Token):
    if tok is None:
        return None
    if '\n' in tok.text:
        return None
    if past_tense_verb(tok):
        return False
    while not isRoot(tok):
        if past_tense_verb(tok.head):
            return False
        elif tensed_clause(tok.head):
            for item in tok.head.subtree:
                if item.dep_ == 'aux' \
                   and item.text.lower() in ['will',
                                             'would',
                                             'shall',
                                             'should',
                                             'can',
                                             'could',
                                             'may',
                                             'might',
                                             'must']:
                    return True
            return False
        if tok.head is not None \
           and tok != tok.head:
            tok = tok.head
        else:
            return False
    for item in tok.head.subtree:
        if item.dep_ == 'aux' \
           and item.text.lower() in ['will',
                                     'would',
                                     'shall',
                                     'should',
                                     'can',
                                     'could',
                                     'may',
                                     'might',
                                     'must']:
            return True
    if isRoot(tok) and 'VerbForm=Inf' in str(tok.morph):
        return False
    return False


def negativePredicate(item):
    """
     This function identifies lexical predicates that function as
     equivalent to negation when combined with other elements. This
     list may not be complete -- to double check later.
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


def isAttributeNoun(item):
    """
     This function identifies nouns that take of complements,
     where the head noun only has a quantifying function. We
     want to allow complements of these nouns to be in the
     scope of negation.
    """
    if item.lemma_ in ['lack',
                       'absence',
                       'shortage',
                       'abundance']:
        return True
    return False


def emptyHeadWord(tok):
    if tok.lemma in ['less',
                     'more',
                     'most',
                     'many',
                     'few',
                     'all',
                     'some',
                     'none',
                     'several',
                     'that',
                     'those',
                     'one',
                     'two',
                     'three',
                     'four',
                     'five',
                     'six',
                     'seven',
                     'eight',
                     'nine',
                     'ten',
                     'part',
                     'portion',
                     'rest',
                     'remnant',
                     'section',
                     'segment']:
        return True
    if tok.tag_ == 'CD':
        return True


def getCompWords():
    return ['than', 'of']


def getLightVerbs():
    return ["have",
            "make",
            "give",
            "present",
            "take",
            "adopt",
            "accept",
            "defend",
            "support",
            "maintain",
            "express"]


def getRoots(doc):
    """
    This function returns the sentence roots of the spaCY dependency tree.
    """
    roots = []
    for tok in doc:
        if isRoot(tok):
            roots.append(tok)
    return roots


def getRoot(token):
    """
    This function returns the sentence root for the current token.
    """
    if isRoot(token):
        return token
    if token.dep_ == '':
        return token
    if token.head is None:
        return token
    return getRoot(token.head)


def isRoot(token):
    if token == token.head \
       or token.dep_ == 'ROOT':
        return True
    elif (token.dep_ == 'conj'
          and token.head == token.head.head):
        return True
    else:
        return False


def rootTree(token, start, end):
    if token.i < start:
        start = token.i
    if token.i > end:
        end = token.i
    for child in token.children:
        if isRoot(child) \
           and tensed_clause(child):
            break
        start, end = rootTree(child, start, end)
    return start, end


def getTensedVerbHead(token):
    if isRoot(token):
        return token

    if token.text.lower() == 'be' \
       and 'MD' in [child.tag_ for child in token.children]:
        return token

    if token.morph is not None \
       and 'PunctSide=Ini' in str(token.morph) \
       and isRoot(token):
        if token.i + 1 < len(token.doc) \
           and token.nbor(1) is not None:
            if token.nbor(1) is None:
                return token
            return getTensedVerbHead(token.nbor(1))
    if token.tag_ in ['VBD', 'VBZ', 'MD']:
        return token
    if token.pos_ == 'VERB':
        if isRoot(token) and 'VerbForm=Inf' in str(token.morph):
            return token
        if (token.dep_ == 'conj'
            or token.tag_ in ['VBG', 'VBN']
            or ('TO' in [item.tag_ for item in token.children])
                and not isRoot(token)):
            if token.head is None:
                return token
            return getTensedVerbHead(token.head)
        if 'Tense=Past' in str(token.morph):
            return token
        elif 'Tense=Pres' in str(token.morph):
            return token
        elif 'am' in [item.text.lower() for item in token.children]:
            return token
        elif 'are' in [item.text.lower() for item in token.children]:
            return token
        elif 'was' in [item.text.lower() for item in token.children]:
            return token
        elif 'were' in [item.text.lower() for item in token.children]:
            return token
        elif 'do' in [item.text.lower() for item in token.children]:
            return token
        elif 'does' in [item.text.lower() for item in token.children]:
            return token
        elif 'did' in [item.text.lower() for item in token.children]:
            return token
        elif 'have' in [item.text.lower() for item in token.children]:
            return token
        elif 'has' in [item.text.lower() for item in token.children]:
            return token
        elif 'had' in [item.text.lower() for item in token.children]:
            return token
        elif 'can' in [item.text.lower() for item in token.children]:
            return token
        elif 'could' in [item.text.lower() for item in token.children]:
            return token
        elif 'will' in [item.text.lower() for item in token.children]:
            return token
        elif 'would' in [item.text.lower() for item in token.children]:
            return token
        elif 'may' in [item.text.lower() for item in token.children]:
            return token
        elif 'might' in [item.text.lower() for item in token.children]:
            return token
        elif 'must' in [item.text.lower() for item in token.children]:
            return token
        elif '\'d' in [item.text.lower() for item in token.children]:
            return token
        elif '\'s' in [item.text.lower() for item
                       in token.children if item.dep_ == 'aux']:
            return token
        elif '\'ve' in [item.text.lower() for item in token.children]:
            return token
        elif '\'ll' in [item.text.lower() for item in token.children]:
            return token
        elif '’d' in [item.text.lower() for item in token.children]:
            return token
        elif '’s' in [item.text.lower() for item
                      in token.children if item.dep_ == 'aux']:
            return token
        elif '’ve' in [item.text.lower() for item in token.children]:
            return token
        elif '’ll' in [item.text.lower() for item in token.children]:
            return token
        else:
            if token.head is None:
                return token
            return getTensedVerbHead(token.head)
    elif isRoot(token):
        return None
    else:
        if token.head is None:
            return token
        return getTensedVerbHead(token.head)


def getSubject(tok: Token):
    for child in tok.children:
        if child.tag_ != '_SP':
            if child.dep_ == 'nsubj' \
               or child.dep_ == 'nsubjpass' \
               or child.dep_ == 'poss' \
               or child.dep_ == 'csubj' \
               or child.dep_ == 'csubjpass' \
               or child.dep_ == 'attr':
                return child
    return None


def getActiveSubject(tok: Token):
    for child in tok.children:
        if child.tag_ != '_SP':
            if child.dep_ == 'nsubj' \
               or child.dep_ == 'poss' \
               or child.dep_ == 'csubj':
                return child
    return None


def getPassiveSubject(tok: Token):
    for child in tok.children:
        if child.tag_ != '_SP':
            if child.dep_ == 'nsubjpass' \
               or child.dep_ == 'poss' \
               or child.dep_ == 'csubjpass':
                return child
    return None


def getObject(tok: Token):
    for child in tok.children:
        if child.tag_ != '_SP':
            if child.dep_ == 'dobj':
                return child
    return None


def quotationMark(token: Token):
    if token.tag_ not in ['-LRB-', '-RRB-']:
        if 'Ini' in token.morph.get('PunctSide'):
            return True
        elif 'Fin' in token.morph.get('PunctSide'):
            return True
        elif token.text in ['"', '"', "'", '“', '”', "''", '``']:
            return True
    return False


def getLogicalObject(tok: Token):
    for child in tok.children:
        try:
            if child.dep_ == 'dobj' \
               or (child.dep_ == 'pobj'
                   and child.lemma_ == 'of') \
               or child.dep_ == 'nsubjpass':
                return child
            if child.dep_ == 'auxpass' \
               and tok._.has_governing_subject:
                return tok.doc[tok._.governing_subject]
            if child.dep_ == 'ccomp' \
               and not tensed_clause(child) \
               and getSubject(child) is not None:
                return getSubject(child)
        except Exception as e:
            print('getlogicalobject', e)
    return None


def getDative(tok: Token):
    for child in tok.children:
        if child.dep_ == 'iobj' \
           or (child.dep_ == 'dative'
               and child.tag_ != 'IN'):
            return child
        elif (child.dep_ == 'dative'
              or (child.dep_ == 'prep'
                  and child.lemma_ == 'to')
              or (child.dep_ == 'prep'
                  and child.lemma_ == 'for')):
            return getDative(child)
        elif child.dep_ == 'pobj':
            return child
    return None


def getPrepObject(tok: Token, tlist):
    for child in tok.children:
        if child.dep_ == 'prep' and child.text.lower() in tlist:
            return getPrepObject(child, tlist)
        elif child.dep_ == 'pobj' and tok.text.lower() in tlist:
            return child
    return None


def scanForAnimatePotentialAntecedents(doc,
                                       loc,
                                       antecedentlocs,
                                       allowDefault=False):
    pos = loc-1
    tok = doc[loc]
    blockedLocs = []
    blockedLex = []
    altAntecedents = []
    for loc in antecedentlocs:
        # don't exclude proper nouns as potential antecedents
        # as they are probably right even if the other co-antecedents
        # are wrong
        if not doc[loc]._.animate:
            blockedLocs.append(loc)
            blockedLex.append(doc[loc].text.lower())
        else:
            altAntecedents.append(loc)
    while pos > 0:
        try:
            if doc[pos].pos_ in ['PRON']:
                Resolution = doc._.coref_chains.resolve(doc[pos])
                if Resolution is not None \
                   and len(Resolution) > 0:
                    resolve = []
                    for item in Resolution:
                        if item._.animate and item.i not in altAntecedents:
                            return [item.i]

                # Let's not ignore a perfectly plausible c-commanding
                # potential antecedent if it happens to be there ...
                if doc[loc] in doc[pos].head.subtree \
                   and doc[pos]._.animate \
                   and (getTensedVerbHead(doc[loc])
                        != getTensedVerbHead(doc[pos])) \
                   and ('Number=Plur' in str(doc[pos].morph)
                        or doc[pos].text.lower()
                        in ['who', 'whom', 'whoever']):
                    return [doc[pos].i]

            elif (doc[pos].pos_ in ['NOUN', 'PROPN']
                  and doc[pos]._.animate
                  and pos not in antecedentlocs
                  and pos not in blockedLocs
                  and doc[pos].text.lower() not in blockedLex):
                if pos not in altAntecedents:
                    altAntecedents.append(pos)

                    # if we have a plural antecedent we don't need
                    # another antecedent to get a plural ...
                    if doc[pos]._.animate \
                       and 'Number=Plur' in str(doc[pos].morph):
                        return [doc[pos].i]
            if len(altAntecedents) > 1:
                if doc[altAntecedents[0]].text.capitalize() == \
                   doc[altAntecedents[1]].text.capitalize():
                    altAntecedents.pop(1)
            if len(altAntecedents) > 1:
                return altAntecedents
        except Exception as e:
            pos -= 1
            print('error in scan', e)
            continue
        pos -= 1
    return [tok.i]


def ResolveReference(tok: Token, doc: Doc):

    # if coreference is turned off, stick with the
    # current word as reference
    if doc._.coref_chains is None:
        return [tok.i]

    # We need to start BERT process that will give us word
    # probabilities in context when we need them. Right now
    # we need that only for checking coreferee results for
    # third person pronouns (which tend to be driven by
    # syntactic considerations even when semantics clearly
    # overrides syntax, e.g., when the antecedent is animate
    # but not in a syntactically dominant location.) But we
    # may need it elsewhere, too, in future. It's a good
    # functionality to have.

    # Right now: have to start the BERT process independently
    # so I've set up the code to set self.wspc to None if it
    # can't find the server.

    # TO-DO: start as a subprocess and use subprocess
    # communication protocols instead of websocket.

    wspc = None
    try:
        wspc = WordseqProbClient()
    except Exception as e:
        print('failed to connect to word \
               sequence probability server\n', e)
    Resolution = doc._.coref_chains.resolve(tok)
    doclist = []

    if tok.text.lower() not in ['they',
                                'them',
                                'their',
                                'theirs',
                                'themselves']:
        if Resolution is not None:
            for item in Resolution:
                doclist.append(item.i)
    else:
        try:
            anim = True
            if Resolution is not None:
                for item in Resolution:
                    if not item._.animate:
                        anim = False
                    else:
                        doclist.append(item.i)
            if not anim:
                for item in doclist:
                    start = tok.i - 4
                    end = tok.i + 4
                    if start < 0:
                        start = 0
                    if end + 1 > len(doc):
                        end = len(doc)
                    if tok.text.lower() in ['their', 'theirs']:
                        res1 = wspc.send(['its',
                                          doc[start:tok.i].text,
                                          doc[tok.i+1:end].text])

                    else:
                        res1 = wspc.send(['things',
                                          doc[start:tok.i].text,
                                          doc[tok.i+1:end].text])

                    if tok.text.lower() in ['their', 'theirs']:
                        res2 = wspc.send(['his',
                                          doc[start:tok.i].text,
                                          doc[tok.i+1:end].text])
                    else:
                        res2 = wspc.send(['people',
                                          doc[start:tok.i].text,
                                          doc[tok.i+1:end].text])

                    if not anim and res2 > res1:
                        antecedentlist = \
                            scanForAnimatePotentialAntecedents(doc,
                                                               tok.i,
                                                               doclist,
                                                               False)
                        if antecedentlist is not None \
                           and len(antecedentlist) > 0:
                            doclist = antecedentlist
                            break

            if len(doclist) == 0:
                antecedentlist = scanForAnimatePotentialAntecedents(
                    doc, tok.i, doclist, True)
                start = tok.i - 4
                if start < 0:
                    start = 0
                end = tok.i + 4
                if end > len(doc):
                    end = len(doc)
                if antecedentlist is not None \
                   and len(antecedentlist) > 0:
                    doclist = antecedentlist
        except Exception as e:
            print('word sequence probability server \
                   is not running.', e)
            if Resolution is not None:
                for item in Resolution:
                    doclist.append(item.i)

    if tok._.vwp_speaker is not None:
        doclist = []
        for item in tok._.vwp_speaker:
            if doc[item].pos_ != 'PRON' \
               and doc[item].lemma_ != 'mine':
                doclist.append(item)
        if len(doclist) == 0:
            doclist.append(tok.i)
        return doclist

    if tok._.vwp_addressee is not None:
        doclist = []
        for item in tok._.vwp_addressee:
            if doc[item].pos_ != 'PRON':
                doclist.append(item)
        if len(doclist) == 0:
            doclist.append(tok.i)
        return doclist

    if len(doclist) > 0:
        return doclist
    else:
        return [tok.i]


reflexives = ['myself',
              'ourselves',
              'yourself',
              'yourselves',
              'himself',
              'herself',
              'itself',
              'themself',
              'themselves']


def getDistinctClauseReferences(tok: Token, hdoc: Doc):
    referenceList = []
    if tok.dep_ in ['nsubj',
                    'nsubjpass',
                    'dobj',
                    'dative'] \
       and tok.text.lower() not in reflexives:
        referenceList.append(tok.i)
        references = ResolveReference(tok, hdoc)
        for reference in references:
            referenceList.append(reference)
    for child in tok.children:
        if child.dep_ in ['nsubj',
                          'nsubjpass',
                          'dobj',
                          'dative'] \
           and child.text.lower() not in reflexives:
            references = ResolveReference(child, hdoc)
            for reference in references:
                if reference not in referenceList:
                    referenceList.append(reference)
        if child.dep_ == 'prep':
            for grandchild in child.children:
                if grandchild.dep_ == 'pobj' \
                   and grandchild.text.lower() not in reflexives:
                    references = ResolveReference(grandchild, hdoc)
                    for reference in references:
                        if reference not in referenceList:
                            referenceList.append(reference)
                elif grandchild.dep_ == 'prep':
                    for ggrandchild in grandchild.children:
                        if ggrandchild.dep_ == 'pobj' \
                           and not ggrandchild.text.lower() in reflexives:
                            references = ResolveReference(ggrandchild, hdoc)
                            for reference in references:
                                if reference not in referenceList:
                                    referenceList.append(reference)

        if child.dep_ in ['acomp',
                          'ccomp',
                          'pcomp',
                          'xcomp']:
            for grandchild in child.children:
                if grandchild.dep_ in ['nsubj', 'nsubjpass'] \
                   and grandchild.text.lower() not in reflexives:
                    references = ResolveReference(grandchild, hdoc)
                    for reference in references:
                        if reference not in referenceList:
                            referenceList.append(reference)
    return referenceList


def containsDistinctReference(tok1: Token, tok2: Token, hdoc: Doc):
    referenceList = getDistinctClauseReferences(tok1, hdoc)
    references = ResolveReference(tok2, hdoc)
    for reference in references:
        if reference in referenceList:
            return True
    return False


def getLinkedNodes(tok: Token):
    linkedList = []
    if tok._.has_governing_subject:
        linkedList.append(tok._.governing_subject)
    if tok.head.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV']:
        linkedList.append(tok.head.i)
    for child in tok.children:
        if child.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV']:
            linkedList.append(child.i)
        else:
            for grandchild in child.children:
                if grandchild.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV']:
                    linkedList.append(grandchild.i)
    return linkedList


first_person_pronouns = ['i',
                         'me',
                         'my',
                         'mine',
                         'myself',
                         'we',
                         'us',
                         'our',
                         'ours',
                         'ourselves']


def all_zeros(a):
    """
     Detect zero vectors that are problematic for agglomerative clustering
    """

    return not np.any(a)


second_person_pronouns = ['you',
                          'your',
                          'yours',
                          'yourself',
                          'yourselves',
                          'u']


def definite(tok: Token):
    for child in tok.subtree:
        if child.text.lower() == 'the':
            return True
        if child.dep_ == 'prp$':
            return True
        if child.dep_ == 'wp$':
            return True
        if child.dep_ == 'poss':
            return True
        break


def match_related_form(token, wordset):
    relatedForms = list(
        np.unique(
            [lemma.derivationally_related_forms()[0].name()
             for lemma in wn.lemmas(token.lemma_)
             if len(lemma.derivationally_related_forms()) > 0]))
    if token.lemma_ not in relatedForms:
        relatedForms.append(token.lemma_)
    if token._.root not in relatedForms:
        relatedForms.append(token._.root)
    for form in relatedForms:
        if form in wordset:
            return True
    return False


def c_command(token1, token2):
    if token2 in token1.head.subtree:
        return True
    else:
        return False


def is_definite_nominal(token):
    if not token.pos_ == 'NOUN' \
       or token.tag_ == 'VBG':
        return False
    for child in token.children:
        if child.dep_ == 'det':
            if child.lemma_ in ['the',
                                'this',
                                'that',
                                'these',
                                'those']:
                return True
        break
    return False


core_temporal_preps = ['in',
                       'on',
                       'over',
                       'upon',
                       'at',
                       'before',
                       'after',
                       'during',
                       'since']

function_word_tags = ['TO',
                      'MD',
                      'IN',
                      'SCONJ',
                      'WRB',
                      'WDT',
                      'WP',
                      'WP$',
                      'EX',
                      'ADP',
                      'JJR',
                      'JJS',
                      'RBR',
                      'RBS']


def temporalPhrase(tok: Token):

    # special case for misparse of phrases like 'during
    # our daily commute to work'
    if tok.pos_ == 'VERB':
        for child in tok.children:
            if child.dep_ == 'mark' \
               and child.lemma_ in ['during']:
                scope = []
                for gchild in child.subtree:
                    scope.append(gchild.i)
                return tok.sent.start, \
                    scope
            break

    if tok.head.pos_ == 'VERB' \
       or tok.head.pos_ == 'AUX':
        if tok.dep_ == 'advmod' \
           and isRoot(tok) \
           and tok.lemma_ in ['early',
                              'late',
                              'later',
                              'earlier',
                              'soon',
                              'ago',
                              'past',
                              'since',
                              'before',
                              'after',
                              'beforehand',
                              'afterward',
                              'afterwards']:
            return tok.sent.start, \
                   [sub.i for sub in tok.subtree]

        if tok.dep_ in ['npadvmod',
                        'attr',
                        'nsubj'] \
            and isRoot(tok.head) \
            and (tok.lemma_.lower() in temporalNouns
                 or is_temporal(tok)):
            scope = []
            for sub in tok.subtree:
                if sub.dep_ in ['mark',
                                'aux',
                                'nsubj',
                                'relcl',
                                'acl',
                                'xcomp']:
                    break
                if sub.lemma_ in ['that',
                                  'which',
                                  'when',
                                  'where',
                                  'why',
                                  'how',
                                  'whether',
                                  'if']:
                    break
                scope.append(sub.i)
            return tok.sent.start, scope

        if tok.dep_ in ['prep', 'mark'] \
           and tok.text.lower() in core_temporal_preps:
            if tok.dep_ == 'mark':
                return tok.i, [tok.i]

            for child in tok.children:
                if child.dep_ in ['pobj', 'pcomp']:
                    if is_temporal(child) \
                       or (is_event(tok)
                           and tok.text.lower() != 'in') \
                       or (child.pos_ == 'VERB'
                           and tok.text.lower() != 'in'):
                        scope = []
                        for sub in tok.subtree:
                            if sub.dep_ in ['mark',
                                            'aux',
                                            'nsubj',
                                            'relcl',
                                            'acl',
                                            'xcomp']:
                                break
                            if sub.lemma_ in ['that',
                                              'which',
                                              'when',
                                              'where',
                                              'why',
                                              'how',
                                              'whether',
                                              'if']:
                                break
                            scope.append(sub.i)
                        return tok.sent.start, scope

    return None


time_period = wn.synsets('time_period')
event = wn.synsets('event')
temporalNouns = ['time',
                 'instant',
                 'point',
                 'occasion',
                 'while',
                 'future',
                 'past',
                 'moment',
                 'second',
                 'minute',
                 'hour',
                 'day',
                 'week',
                 'month',
                 'year',
                 'century',
                 'millenium',
                 'january',
                 'february',
                 'march',
                 'april',
                 'may',
                 'june',
                 'july',
                 'august',
                 'september',
                 'october',
                 'november',
                 'december',
                 'monday',
                 'tuesday',
                 'wednesday',
                 'thursday',
                 'friday',
                 'saturday',
                 'sunday',
                 'today',
                 'tomorrow',
                 'yesterday',
                 'noon',
                 'midnight',
                 'o\'clock',
                 'a.m.',
                 'p.m.',
                 'afternoon',
                 'morning',
                 'evening']


def is_temporal(tok: Token):
    if not tok.pos_ == 'NOUN':
        return False
    if tok.lemma_.lower() in temporalNouns:
        return True
    if tok.ent_type_ in ['TIME', 'DATE', 'EVENT']:
        return True
    synsets = wn.synsets(tok.lemma_)
    if len(synsets) > 0:
        hypernyms = set([i for i in
                         synsets[0].closure(
                             lambda s:s.hypernyms())])
        if len(hypernyms) > 0:
            if time_period[0] in hypernyms \
               or time_period[0] == synsets[0]:
                return True
    return False


def is_event(tok: Token):
    if not tok.pos_ == 'NOUN':
        return False
    synsets = wn.synsets(tok.lemma_)
    if len(synsets) > 0:
        hypernyms = set([i for i in
                        synsets[0].closure(
                            lambda s: s.hypernyms())])
        if len(hypernyms) > 0:
            if (event[0] in hypernyms
               or event[0] == synsets[0]):
                for lemma in synsets[0].lemmas():
                    if token.lemma_ == lemma.name():
                        for word in lemma.derivationally_related_forms():
                            if word.synset().pos() == 'v':
                                return True
    return False


content_tags = ['NN',
                'NNS',
                'NNP',
                'NNPS',
                'VB',
                'VBD',
                'VBG',
                'VBN',
                'VBP',
                'VBZ',
                'JJ',
                'JJR',
                'JJS',
                'RB',
                'RBR',
                'RBS',
                'RP',
                'GW',
                'NOUN',
                'PROPN',
                'VERB',
                'ADJ',
                'ADV',
                'CD']

content_pos = ['NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV', 'CD']

major_locative_prepositions = ['to',
                               'from',
                               'in',
                               'on',
                               'at',
                               'upon',
                               'over',
                               'under',
                               'beneath',
                               'beyond',
                               'along',
                               'against',
                               'through',
                               'throughout',
                               'by',
                               'near',
                               'into',
                               'onto',
                               'off',
                               'out']

all_locative_prepositions = ['above',
                             'across',
                             'against',
                             'along',
                             'amid',
                             'amidst',
                             'among',
                             'amongst',
                             'around',
                             'at',
                             'athwart',
                             'atop',
                             'before',
                             'below',
                             'beneath',
                             'beside',
                             'between',
                             'betwixt',
                             'beyond',
                             'down',
                             'from',
                             'in',
                             'inside',
                             'into',
                             'near',
                             'off',
                             'on',
                             'opposite'
                             'out',
                             'outside',
                             'over',
                             'through',
                             'throughout',
                             'to',
                             'toward',
                             'under',
                             'up',
                             'within',
                             'without',
                             'yon',
                             'yonder']

deictics = ['i',
            'me',
            'my',
            'mine',
            'myself',
            'we',
            'us',
            'our',
            'ours',
            'ourselves',
            'you',
            'your',
            'yours',
            'yourself',
            'yourselves',
            'here',
            'there',
            'hither',
            'thither',
            'yonder',
            'yon',
            'now',
            'then',
            'anon',
            'today',
            'tomorrow',
            'yesterday',
            'this',
            'that',
            'these',
            'those'
            ]

adj_noun_or_verb = ['NN',
                    'NNS',
                    'NNP',
                    'NNPS',
                    'VB',
                    'VBD',
                    'VBG',
                    'VBN',
                    'VBP',
                    'VBZ',
                    'JJ',
                    'JJR',
                    'JJS',
                    'RP',
                    'GW',
                    'NOUN',
                    'PROPN',
                    'VERB',
                    'ADJ']

possessive_or_determiner = ['PRP',
                            'PRP$',
                            'WDT',
                            'WP',
                            'WP$',
                            'WRB',
                            'DT']

personal_or_indefinite_pronoun = ['i',
                                  'me',
                                  'my',
                                  'mine',
                                  'we',
                                  'us',
                                  'our',
                                  'ours',
                                  'you',
                                  'your',
                                  'yours',
                                  'he',
                                  'him',
                                  'they',
                                  'them',
                                  'their',
                                  'theirs',
                                  'his',
                                  'she',
                                  'her',
                                  'hers',
                                  'everyone',
                                  'anyone',
                                  'everybody',
                                  'anybody',
                                  'nobody',
                                  'someone',
                                  'somebody',
                                  'myself',
                                  'ourselves',
                                  'yourself',
                                  'yourselves',
                                  'himself',
                                  'herself',
                                  'themselves',
                                  'one',
                                  'oneself',
                                  'oneselves'
                                  'anothers',
                                  'others',
                                  'another',
                                  'some',
                                  'many'
                                  'few',
                                  'none',
                                  'who',
                                  'whom',
                                  'whoever']

loc_sverbs = ['contain',
              'cover',
              'include',
              'occupy']

loc_overbs = ['abandon',
              'approach',
              'clear',
              'depart',
              'inhabit',
              'occupy',
              'empty',
              'enter',
              'escape',
              'exit',
              'fill',
              'leave',
              'near']

locative_adverbs = ['here',
                    'there',
                    'where',
                    'somewhere',
                    'anywhere']

existential_there = 'EX'

prehead_modifiers = ['mark',
                     'nsubj',
                     'nsubjpass',
                     'aux',
                     'neg',
                     'det',
                     'poss']

quantifying_determiners = ['any',
                           'all',
                           'no',
                           'each',
                           'every',
                           'little',
                           'some',
                           'few',
                           'more',
                           'most']

subject_or_object_nom = ['nsubj',
                         'nsubjpass',
                         'dobj']

clausal_complements = ['csubj',
                       'ccomp',
                       'xcomp',
                       'acl',
                       'oprd']

complements = ['csubj',
               'ccomp',
               'nsubj',
               'nsubjpass',
               'dobj',
               'xcomp',
               'acl',
               'oprd',
               'attr',
               'acomp']


def clausal_subject_or_complement(tok):
    if (tok.dep_ in ['xcomp', 'oprd', 'csubj']
        or (tok.dep_ in ['ccomp', 'acl']
            and tensed_clause(tok))):
        return True
    else:
        return False


adjectival_predicates = ['attr', 'oprd', 'acomp', 'amod']


def sylco(word):
    """
    from discussion posted to
    https://stackoverflow.com/questions/46759492/syllable-count-in-python

    Fallback to calculate number of syllables for words that aren't in the
    moby hyphenator lexicon.
    """
    word = word.lower()

    if not alphanum_word(word):
        return None

    # exception_add are words that need extra syllables
    # exception_del are words that need less syllables

    exception_add = ['serious', 'crucial']
    exception_del = ['fortunately', 'unfortunately']

    co_one = ['cool',
              'coach',
              'coat',
              'coal',
              'count',
              'coin',
              'coarse',
              'coup',
              'coif',
              'cook',
              'coign',
              'coiffe',
              'coof',
              'court']
    co_two = ['coapt', 'coed', 'coinci']

    pre_one = ['preach']

    syls = 0  # added syllable number
    disc = 0  # discarded syllable number

    # 1) if letters < 3 : return 1
    if len(word) <= 3:
        syls = 1
        return syls

    # 2) if doesn't end with "ted" or "tes" or "ses" or "ied" or "ies",
    # discard "es" and "ed" at the end. If it has only 1 vowel or 1 set
    # of consecutive vowels, discard. (like "speed", "fled" etc.)

    if word[-2:] == "es" or word[-2:] == "ed":
        doubleAndtripple_1 = len(re.findall(r'[eaoui][eaoui]', word))
        if doubleAndtripple_1 > 1 \
           or len(re.findall(r'[eaoui][^eaoui]', word)) > 1:
            if word[-3:] == "ted" \
               or word[-3:] == "tes" \
               or word[-3:] == "ses" \
               or word[-3:] == "ied" \
               or word[-3:] == "ies":
                pass
        else:
            disc += 1

    # 3) discard trailing "e", except where ending is "le"
    le_except = ['whole',
                 'mobile',
                 'pole',
                 'male',
                 'female',
                 'hale',
                 'pale',
                 'tale',
                 'sale',
                 'aisle',
                 'whale',
                 'while']

    if word[-1:] == "e":
        if word[-2:] == "le" and word not in le_except:
            pass

        else:
            disc += 1

    # 4) check if consecutive vowels exists, triplets or pairs,
    #    count them as one.

    doubleAndtripple = len(re.findall(r'[eaoui][eaoui]', word))
    tripple = len(re.findall(r'[eaoui][eaoui][eaoui]', word))
    disc += doubleAndtripple + tripple

    # 5) count remaining vowels in word.
    numVowels = len(re.findall(r'[eaoui]', word))

    # 6) add one if starts with "mc"
    if word[:2] == "mc":
        syls += 1

    # 7) add one if ends with "y" but is not surrouned by vowel
    if word[-1:] == "y" and word[-2] not in "aeoui":
        syls += 1

    # 8) add one if "y" is surrounded by non-vowels and is
    #    not in the last word.

    for i, j in enumerate(word):
        if j == "y":
            if (i != 0) and (i != len(word) - 1):
                if word[i-1] not in "aeoui" and word[i+1] not in "aeoui":
                    syls += 1

    # 9) if starts with "tri-" or "bi-" and is followed by a vowel,
    #    add one.

    if word[:3] == "tri" and word[3] in "aeoui":
        syls += 1

    if word[:2] == "bi" and word[2] in "aeoui":
        syls += 1

    # 10) if ends with "-ian", should be counted as two syllables,
    #  except for "-tian" and "-cian"

    if word[-3:] == "ian" and (word[-4:] != "cian" or word[-4:] != "tian"):
        if word[-4:] == "cian" or word[-4:] == "tian":
            pass
        else:
            syls += 1

    # 11) if starts with "co-" and is followed by a vowel, check if exists
    # in the double syllable dictionary, if not, check if in single
    # dictionary and act accordingly.

    if word[:2] == "co" and word[2] in 'eaoui':

        if word[:4] in co_two or word[:5] in co_two or word[:6] in co_two:
            syls += 1
        elif (word[:4] in co_one
              or word[:5] in co_one
              or word[:6] in co_one):
            pass
        else:
            syls += 1

    # 12) if starts with "pre-" and is followed by a vowel, check if
    # exists in the double syllable dictionary, if not, check if in
    # single dictionary and act accordingly.

    if word[:3] == "pre" and word[3] in 'eaoui':
        if word[:6] in pre_one:
            pass
        else:
            syls += 1

    # 13) check for "-n't" and cross match with dictionary to add syllable.
    negative = ["doesn't", "isn't", "shouldn't", "couldn't", "wouldn't"]

    if word[-3:] == "n't":
        if word in negative:
            syls += 1
        else:
            pass

    # 14) Handling the exceptional words.

    if word in exception_del:
        disc += 1

    if word in exception_add:
        syls += 1

    sylcount = numVowels - disc + syls

    if sylcount == 0:
        sylcount = 1
    # calculate the output
    return sylcount


def alphanum_word(word: str):
    if not re.match('[-A-Za-z0-9\'.]', word) or re.match('[-\'.]+', word):
        return False
    else:
        return True
 
def is_float(str):
    try:
        float(str)
        return True
    except ValueError:
        return False 

def newSpanEntry(name, left, right, hdoc, value):
    '''
        Create an entry in the format used for span indicator
        values by the AWE_Information function. The offset and
        length attributes support finding the exact span in
        the input text. The startToken and endToken attributes
        support finding the correct tokens in the Spacy parse
        tree. The value field contains whatever data we want
        to associate with the span, which depends on the
        specific indicator.
    '''
    entry = {}
    entry['name'] = name
    entry['offset'] = hdoc[left].idx
    entry['startToken'] = left
    entry['endToken'] = right
    entry['length'] = hdoc[right].idx \
        + len(hdoc[right].text_with_ws) \
        - hdoc[left].idx
    entry['value'] = value
    entry['text'] = hdoc[left:right+1].text
    return entry

def newTokenEntry(name, token, value):
    '''
       Create an entry in the format used for token indicator 
       values by the AWE_Information function. The value needs
       to be supplied either by a built-in function or a spacy
       extension attribute. The offset and length attributes
       enable us to find the exact character sequenc ethat
       corresponds to this token in the original text string.
       The tokenIdx attribute is the index into the Spacy
       token list. The value function is whatever data we
       wish to associate with this function, which depends on
       the specific indicator.
    '''
    entry = {}
    entry['text'] = token.text_with_ws
    entry['offset'] = token.idx
    entry['tokenIdx'] = token.i
    entry['length'] = len(token.text_with_ws)
    entry['name'] = name
    entry['value'] = None

    if name in built_in_attributes:
        entry['value'] = eval('token.' + name)

    #######################################
    # Create entries for named extensions #
    # Note that we assume extensions are  #
    # attributes not functions.           #
    # Code will break if the indicator    #
    # name is a function extended         #
    # attribute.                          #
    # TBD: put security check in for this #
    #######################################
    elif token.has_extension(name):
        entry['value'] = eval('token._.' + name)
    else:
        raise AWE_Workbench_Error('Invalid indicator '
            + name)

    return entry

def createSpanInfo(indicator, document):
    '''
        Create records for span data in the format used
        by the AWE_Info function
    '''
    baseInfo = []
    entry = {}
    # Create span records for sentence spans
    if indicator == 'sents':
        for sent in document.sents:
            entry = \
                newSpanEntry('sents',
                    sent.start,
                    sent.end-1,
                    document,
                    'sentence')
            baseInfo.append(entry)

    # Create span records for other kinds of spans
    # created in a parser module. Those functions
    # are responsible to make sure the outputs
    # are in the correct format.
    elif indicator in docspan_extensions:
        baseInfo = eval('document._.' + indicator)

    # Create span records for paragraphs (delimiter_\n)
    # and other spans identified by delimiting characters
    elif indicator.startswith('delimiter_'):
        delimiter = ''.join(indicator[10:])
        segmentNo = 0
        currentStart = 0
        currentEnd = 0
        for token in document:
            currentEnd = token.i
            if delimiter in token.text:
                entry = \
                    newSpanEntry(indicator,
                        currentStart,
                        currentEnd,
                        document,
                        delimiter)
                baseInfo.append(entry)
                currentStart = token.i
                segmentNo += 1
        if currentEnd > currentStart:
            entry = \
                newSpanEntry(indicator,
                    currentStart,
                    token.i,
                    document,
                    segmentNo)
            baseInfo.append(entry)
    else:
        raise AWE_Workbench_Error(
            'Invalid indicator '
            + indicator)

    return baseInfo

def applyTokenFilters(token, entry, filters):
    '''
        Given an entry in the format used to describe
        indicator values for tokens by the AWE_Info function,
        check if that entry passes the specified filters.
    '''
    filterEntry = False
    for (function, returnValues) in filters:
        if type(filters) == list and len(filters)>0:
            for (function, returnValues) in filters:
                for returnValue in returnValues:

                    # Direct comparison with the returnValue
                    if function in ['==',
                                    '>',
                                    '<',
                                    '>=',
                                '    <='] \
                       and type(entry['value']) \
                          in [int, float, str]:
                        if not eval(entry['value'] \
                           + function \
                           + returnValue):
                            return True

                    # The returnValue specifies a boolean value
                    elif type(entry['value']) == bool \
                       and returnValue == 'True':
                        if not entry['value']:
                            return True

                    elif type(entry['value']) == bool \
                       and returnValue == 'False':
                         if entry['value']:
                             return True

                    # Negation
                    elif function == 'not' \
                       and type(entry['value']) == bool:
                        if entry['value']:
                            return True

                    # Spacy built-in boolean token flags
                    elif function in built_in_flags:
                        if returnValue == 'True':
                            if not eval('token.' + function):
                                return True
                        elif returnValue == 'False':
                            if eval('token.' + function):
                                return True
                        else:
                            raise AWE_Workbench_Error(
                                'Invalid selection value '
                                + returnValue)

                    # Spacy built-in string functions
                    elif function in built_in_string_functions:
                        if eval('token.' + function) == returnValue:
                            return False
                        else:
                            filterEntry = True

                    # Spacy extension attributes, if true boolean
                    # flags or numeric (non-zero) values
                    elif token.has_extension(function):
                        if eval('token._.' + function) is not None \
                           and eval('token._.' + function) \
                              in [True, False]:
                             if returnValue == 'True':
                                 if not eval('token._.'
                                             + function):
                                     return False
                             elif returnValue == 'False':
                                 if eval('token._.' 
                                         + function):
                                     return False
                             else:
                                 filterEntry = True

                    else:
                        raise AWE_Workbench_Error(
                            'Invalid filter ' + function)
                if filterEntry:
                    return True
    return False


def applySpanTransformations(transformations, baseInfo):
    '''
       Apply transformation to span entries in the format used
       by the AWE_Info function, in the order listed
    '''
    for transformation in transformations:
        # security check
        if not re.match('[A-Za-z0-9_]+', transformation):
            raise AWE_Workbench_Error(
                'Invalid transformation'
                + transformation)                   

        newInfo = []
        for entry in baseInfo:
            newEntry = entry
            if transformation == 'text':
                 newEntry['value'] = entry['text']
                 newEntry['name'] = 'text_' \
                     + entry['name']

            elif transformation == 'len' \
               and type(entry['value']) in [str, list]:
                newEntry['value'] = entry['length']
                newEntry['name'] = 'clen_' \
                    + entry['name']

            if transformation == 'tokenlen' \
               and type(entry['value']) == str:
                newEntry['value'] = 1 \
                    + entry['endToken'] \
                    - entry['startToken']
                newEntry['name'] = 'tlen_' \
                     + entry['name']

            newInfo.append(newEntry)
        baseInfo = newInfo
    return baseInfo

    
def applyTokenTransformations(entry, token, transformations):
    '''
       Given an entry with information about a token in the format
       used by the AWE_Info function, transform the value
       as specified by the transformations list
    '''
    for transformation in transformations:
    
        # security check
        if not re.match('[A-Za-z0-9_]+', transformation):
            raise AWE_Workbench_Error(
                'Invalid transformation'
                + transformation)                   

        if transformation == 'text':
            entry['value'] = entry['text']
            entry['name'] = 'text_' + entry['name']                  

        elif transformation == 'len' \
           and type(entry['value']) in [str, list]:
            entry['value'] = len(entry['value'])
            entry['name'] = 'len_' + entry['name']

        elif transformation in built_in_flags:
            entry['value'] = eval('token.'
                                  + transformation)
            entry['name'] = transformation \
                + '_' + entry['name']

        elif transformation in ['log', 'sqrt']:
            if entry['value'] is not None \
               and is_float(entry['value']) \
               and not (transformation == 'sqrt'
                        and entry['value'] < 0):

                if transformation == 'log':
                    entry['value'] = math.log(entry['value'])
                    entry['name'] = 'log_' + entry['name']

                elif transformation == 'sqrt':
                    entry['value'] = math.sqrt(entry['value'])
                    entry['name'] = 'sqrt_' + entry['name']

                elif transformation == 'log' \
                   and entry['value'] is not None:                
                    raise AWE_Workbench_Error(
                        'Cannot log non numeric data')

                elif transformation == 'sqrt' \
                   and entry['value'] is not None:                
                     raise AWE_Workbench_Error(
                         'Cannot take the square'
                         + ' root of non numeric data')
    return entry


def applySummaryFunction(info, baseInfo, summaryType, document):
    '''
        Given a matrix of information about indicator values
        for the tokens in a document, apply a summary function
        and return the resulting summary information to the main
        AWE_Info function
    '''    

    # Security check
    if summaryType != '' and summaryType is not None \
       and not re.match('[A-Za-z0-9_]+', summaryType):
        raise AWE_Workbench_Error('Invalid summary function '
            + summaryType)

    if len(info) == 0 \
       and (summaryType is None
            or summaryType==''):
        return json.dumps({})

    # Counts of unique values
    if summaryType == "counts":
        if len(info)==0:
            return json.dumps({})
        output = {}
        summary = info['value'].value_counts()
        for i, value in enumerate(summary):
            entry = {}
            category = summary.index[i]
            if type(summary.index[i]) != str:
                category = summary.index[i]
            if type(category) == list:
                category = json.dumps(category)
            output[category] = int(value)
        return json.dumps(output)

    # total number of entries in info
    elif summaryType == "total":
        return len(info)

    # list of unique values
    elif summaryType == "uniq":
        if len(info)==0:
            return json.dumps({})
        output = []
        summary = info['value'].value_counts()
        for i, value in enumerate(summary):
            category = summary.index[i]
            if type(summary.index[i]) != str:
                category = json.dumps(summary.index[i])
            if category not in output:
                output.append(category)
        return output

    # Total number of unique values
    elif summaryType == "totaluniq":
        if len(info)==0:
            return 0
        output = []
        summary = info['value'].value_counts()
        for i, value in enumerate(summary):
            category = summary.index[i]
            if type(summary.index[i]) != str:
                category = json.dumps(summary.index[i])
            if category not in output:
                output.append(category)
        return len(output)

    # Proportion or percent
    elif summaryType in ["proportion", "percent"]:
        if len(info)==0:
            return None
        total = 0
        for index, row in info.iterrows():
            if row['value']:
                total += 1
        if len(info) > 0:
            if summaryType == "proportion":
                return total/len(document)
            else:
                return round(100*total/len(document))
        else:
            return None
            
    # Mean
    elif summaryType == "mean":
        if len(info)==0:
            return None
        mean = info['value'].mean(axis=0)
        if type(mean) == type(np.int64(0)):
            return int(mean)
        else:
            return float(mean)

    # Median
    elif summaryType == "median":
        if len(info)==0:
            return None
        median = info['value'].median(axis=0)
        if type(median) == type(np.int64(0)):
            return int(median)
        else:
            return float(median)

    # Standard Deviation
    elif summaryType == "stdev":
        if len(info)==0:
            return None
        if len(baseInfo) > 2:
            std = info['value'].std(axis=0)
            if type(std) == type(np.int64(0)):
                return int(std)
            else:
                return float(std)
        else:
            return None

    # Maximum
    elif summaryType == "max":
        if len(info)==0:
            return None
        maxVal = info['value'].max(axis=0)
        if type(maxVal) == type(np.int64(0)):
            return int(maxVal)
        else:
            return float(maxVal)

    # Minimum
    elif summaryType == "min":
        if len(info)==0:
            return None
        minVal = info['value'].min(axis=0)
        if type(minVal) == type(np.int64(0)):
            return int(minVal)
        else:
            return float(minVal)

    # No summary, just the full dataframe
    elif summaryType == '' or summaryType is None:
        if len(info)==0:
            return None
        val = info.T.to_json()
        return val
    else:
        raise AWE_Workbench_Error('Invalid summary function '
            + summaryType)


def AWE_Info(document: Doc,
             infoType='Token',
             indicator='pos_',
             filters=[],
             transformations=[],
             summaryType=None):
    ''' This function provides a general-purpose API for
        obtaining information about indicators annoted on
        the AWE Workbench Spacy parse tree. 
    '''
    try:
        baseInfo = []
        # security check
        if not re.match('[A-Za-z0-9_]+', indicator):
            raise AWE_Workbench_Error(
                'Invalid indicator ' + indicator)                   
    
        if infoType == 'Doc':
            baseInfo = createSpanInfo(indicator,
                                      document)
            baseInfo = applySpanTransformations(transformations,
                                                baseInfo)
        elif infoType == 'Token':
            for token in document:
                entry = newTokenEntry(indicator, token, None)

                filterEntry = False
                if type(filters) == list and len(filters)>0:
                    filterEntry = applyTokenFilters(token,
                                                    entry,
                                                    filters)
                elif filters != []:
                    raise AWE_Workbench_Error('Invalid filter '
                        + str(filters))                   
                if filterEntry:
                    continue
                else:
                    entry = \
                        applyTokenTransformations(entry,
                                                  token,
                                                  transformations)
                    baseInfo.append(entry)
        else:
            raise AWE_Workbench_Error('Invalid indicator type '
                + infoType)                   
        
        info = pd.DataFrame.from_dict(baseInfo)
        return applySummaryFunction(info,
                                    baseInfo,
                                    summaryType,
                                    document)

    except Exception as e:
            print(e)
            raise AWE_Workbench_Error('error in code')
