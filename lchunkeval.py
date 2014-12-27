#!/usr/bin/env python2.7
#coding=utf-8
'''
lchunkeval: Evaluation utility for labeled chunking tasks, such as named entity recognition 
(detecting and classifying names) and lexical semantic analysis (segmenting into 
lexical semantic units and classifying them semantically).

The implemented evaluation measure is based on edit distance.
Basic edit operations are used to transform the predicted analysis 
into the gold analysis; the sequence of such edits is called the edit script. 
Costs are assigned to individual edit operations, and the cost of the script 
is the sum of the component costs. Thus the edit distance between two analyses 
is the cost of a least expensive edit script from one to the other, 
found via Uniform Cost Search (Dijkstra's algorithm).

@author: Nathan Schneider (nschneid@inf.ed.ac.uk)
@since: 2014-12-23
'''
from __future__ import print_function, division
from collections import Counter
import heapq, itertools, sys



LABELS = set()

def lbl(tag): return tag[2:]
def label(tag, default=''): return lbl(tag) if len(tag)>1 else default

def isBegin(tag): return tag.upper().startswith('B')
def isInside(tag): return tag.upper()[0] in ('I', ) # TODO: strong/weak
def isOutside(tag): return tag.upper().startswith('O')
def isInGap(tag): return tag[0].islower() # TODO: check with strong/weak

def toBegin(nonOutsideTag): return nonOutsideTag[0].replace('I','B').replace('i','b')+nonOutsideTag[1:] # TODO: strength
def toInside(nonOutsideTag): return nonOutsideTag[0].replace('B','I').replace('b','i')+nonOutsideTag[1:]    # TODO: strength
def toOutside(tag): return 'O' if not isInGap(tag) else 'o'
def toInGap(tag): return tag[0].lower()+tag[1:]
def toOutOfGap(tag): return tag[0].upper()+tag[1:]
def attachLabel(flag,l): return flag+'-'+l

def isValidTag(tag):
    if len(tag)>1 and tag[1]!='-': return False
    if len(tag)==2: return False
    if not (isBegin(tag) or isInside(tag) or isOutside(tag)):
        return False
    return True

def isValidBigram(tag1,tag2):   # None = sequence boundary
    if tag1 is None:
        return tag2 is not None and not isInGap(tag2) and not isInside(tag2)
    if tag2 is None:
        return tag1 is not None and not isInGap(tag1)
    if isInGap(tag1)==isInGap(tag2):
        if isOutside(tag1) and isInside(tag2): return False
        if isInside(tag2) and label(tag1)!=label(tag2): return False # TODO: revisit (maybe don't require label on I- tags)
    elif isInGap(tag1) and not isInGap(tag2):
        if not isInside(tag2): return False
    elif isInGap(tag2) and not isInGap(tag1):
        if isOutside(tag1): return False
    # note that this function doesn't check non-local label dependencies across gaps
    return True

def isValidTagging(sequence):
    '''
    >>> isValidTagging(['B-PER', 'I-PER'])
    True
    >>> isValidTagging(['O', 'O', 'O', 'B-PER'])
    True
    >>> isValidTagging(['O', 'O', 'O', 'I-PER'])
    False
    >>> isValidTagging(['I-ORG', 'O', 'O', 'B-PER'])
    False
    >>> isValidTagging(['O', 'O', 'O', 'b-PER'])
    False
    >>> isValidTagging(['O', 'B-PER', 'O', 'I-PER'])
    False
    >>> isValidTagging(['O', 'B-PER', 'o', 'I-PER'])
    True
    >>> isValidTagging(['O', 'B-PER', 'o', 'I-ORG'])
    False
    >>> isValidTagging(['O', 'B-PER', 'b-ORG', 'I-PER'])
    True
    >>> isValidTagging(['O', 'B-PER', 'b-ORG', 'I-ORG'])
    False
    >>> isValidTagging(['B-PER', 'b-ORG', 'i-ORG', 'I-PER'])
    True
    >>> isValidTagging(['B-PER', 'b-ORG', 'i-ORG', 'i-ORG', 'I-PER', 'I-PER'])
    True
    '''
    lastNotInGap = None
    for t in sequence:
        if not isValidTag(t): 
            #assert False,('bad tag',t)
            return False
        if not isInGap(t):
            if isInside(t): # check non-local label dependencies across gaps
                if lastNotInGap and label(lastNotInGap)!=label(t): return False
            lastNotInGap = t
    
    for t1,t2 in zip(itertools.chain([None],sequence), itertools.chain(sequence,[None])):
        if not isValidBigram(t1, t2):
            #assert False,('bad bigram',t1,t2)
            return False
    
    return True

def chunks(tags):   # TODO: option: does O count as a singleton chunk?
    '''
    >>> list(chunks(['O', 'B-evt', 'o', 'b-PER', 'I', 'I', 'B-PER', 'O', 'B-ORG', 'I-ORG'])) \
        #             0    1        2    3        4    5    6        7    8        9
    [(3,), (1, 4, 5), (6,), (8, 9)]
    '''
    ochk = []
    ichk = None
    for i,t in enumerate(tags):
        if isInGap(t):
            if ichk is None:
                assert not isInside(t)
            else:
                if isInside(t):
                    ichk.append(i)
                elif ichk:
                    yield tuple(ichk)
                    ichk = []
            if isBegin(t):
                ichk = [i]
        else:
            if ichk: yield tuple(ichk)  # post-gap
            ichk = None
            if isInside(t):
                ochk.append(i)
            elif ochk:
                yield tuple(ochk)
                ochk = []
            if isBegin(t):
                ochk = [i]
    assert ichk is None
    if ochk: yield tuple(ochk)


C_DEL = C_INS = 2
C_REL = 1
C_SPL = C_MRG = 1
C_NAR = C_WID = 1

def legal_edits(script, out_tags):
    '''
    A generator over edit moves given a script (including the source sequence) 
    and the target sequence. Produces 3-tuples of of the form (edit, result, cost). 
    Each edit is represented as a tuple containing: 
    the name of the operation (see below); the offset(s) indicating
    where in the sequence it applies; and the label(s) of the chunks to which it applies. 
    
    
    Synposis of Edit Operations
    ===========================
    
    - RELABEL changes the label of a single chunk: B-PER (I-PER)* ↔ B-ORG (I-ORG)*
      on the condition that at least one of the tokens in the chunk is labeled with ORG in the output
    - INSERT(DELETE) applies to a single chunk: O (O)* → B-PER (I-PER)*
      on the condition that the DELETEd chunk is present in the source, and the INSERTed 
      chunk is present in the target
    - SPLIT(MERGE) separates into (combines from) 2 adjacent chunks: 
      B-PER I-PER → B-PER B-PER; I-PER I-PER → I-PER B-PER
    - NARROW1LEFT(WIDEN1LEFT) shifts the left boundary of a chunk: B-PER I-PER → O B-PER
    - NARROW1RIGHT: B|I-PER I-PER ] → O  (where "]" means "followed by a non-I tag or the end of the sequence")
    - WIDEN1RIGHT: B-PER O → B-PER I-PER; I-PER O → I-PER I-PER
    
    The above rules also work with gappy expressions, so long as a rule application 
    does not affect tags both inside and outside of a gap. 
    They are associated with constant costs: `C_REL`, `C_INS`, `C_DEL`, `C_SPL`, `C_MRG`, 
    `C_NAR`, and `C_WID`.
    
    TODO: Consider adding INSERT-GAP, DELETE-GAP, WIDEN-GAP-{LEFT,RIGHT}, 
    NARROW-GAP-{LEFT,RIGHT}; STRENGTHEN, WEAKEN
    
    See best_script() documentation for constraints on the application of these edits.
    
    
    Test Cases
    ==========
    
    >>> LABELS |= set(['PER', 'ORG', 'LOC', 'evt'])
    >>> list(legal_edits([(None, ['O'])], ['O']))
    []
    >>> sorted(legal_edits([(None, ['B-evt'])], ['B-LOC']))
    [(('DELETE', (0,), 'evt'), ['O'], 2), 
     (('RELABEL', (0,), 'evt', 'LOC'), ['B-LOC'], 1)]
    >>> sorted(legal_edits([(None, ['B-evt', 'o', 'I-evt'])], ['O', 'B-LOC', 'O']))
    [(('DELETE', (0, 2), 'evt'), ['O', 'O', 'O'], 2), 
     (('INSERT', (1,), 'LOC'), ['B-evt', 'b-LOC', 'I-evt'], 2)]
    >>> list(legal_edits([(None, ['O', 'O', 'O'])], ['B-LOC', 'I-LOC', 'O']))
    [(('INSERT', (0, 1), 'LOC'), ['B-LOC', 'I-LOC', 'O'], 2)]
    >>> list(legal_edits([(None, ['O', 'O', 'O'])], ['B-evt', 'b-LOC', 'I-evt']))
    [(('INSERT', (1,), 'LOC'), ['O', 'B-LOC', 'O'], 2), 
     (('INSERT', (0, 2), 'evt'), ['B-evt', 'o', 'I-evt'], 2)]
    >>> sorted(legal_edits([(None, ['O', 'B-PER', 'o', 'I-PER'])], ['B-evt', 'o', 'I-evt', 'O']))
    [(('DELETE', (1, 3), 'PER'), ['O', 'O', 'O', 'O'], 2), 
     (('WIDEN1LEFT', 1, 'PER'), ['B-PER', 'I-PER', 'o', 'I-PER'], 1)]
    >>> sorted(legal_edits([(None, ['O', 'B-PER', 'o', 'I-PER'])], ['O', 'B-evt', 'b-ORG', 'I-evt']))
    [(('DELETE', (1, 3), 'PER'), ['O', 'O', 'O', 'O'], 2), 
     (('INSERT', (2,), 'ORG'), ['O', 'B-PER', 'b-ORG', 'I-PER'], 2), 
     (('RELABEL', (1, 3), 'PER', 'evt'), ['O', 'B-evt', 'o', 'I-evt'], 1), 
     (('WIDEN1LEFT', 1, 'PER'), ['B-PER', 'I-PER', 'o', 'I-PER'], 1)]
    >>> sorted(legal_edits([(None, ['O', 'B-PER', 'b-LOC', 'I-PER'])], ['O', 'B-evt', 'b-ORG', 'I-evt']))
    [(('DELETE', (1, 3), 'PER'), ['O', 'O', 'B-LOC', 'O'], 2), 
     (('DELETE', (2,), 'LOC'), ['O', 'B-PER', 'o', 'I-PER'], 2),
     (('RELABEL', (1, 3), 'PER', 'evt'), ['O', 'B-evt', 'b-LOC', 'I-evt'], 1),
     (('RELABEL', (2,), 'LOC', 'ORG'), ['O', 'B-PER', 'b-ORG', 'I-PER'], 1),
     (('WIDEN1LEFT', 1, 'PER'), ['B-PER', 'I-PER', 'b-LOC', 'I-PER'], 1)]
    >>> list(e for e in legal_edits([(None, ['O', 'O', 'B-PER', 'O', 'O'])], ['O']*5) if e[0][0]!='RELABEL')
    [(('DELETE', (2,), 'PER'), ['O', 'O', 'O', 'O', 'O'], 2), 
     (('WIDEN1LEFT', 2, 'PER'), ['O', 'B-PER', 'I-PER', 'O', 'O'], 1), 
     (('WIDEN1RIGHT', 2, 'PER'), ['O', 'O', 'B-PER', 'I-PER', 'O'], 1)]
    >>> list(e for e in legal_edits([(None, ['O', 'O', 'B-PER', 'I-PER', 'O', 'O'])], ['B-PER','I-PER','I-PER','O','O','O']) if e[0][0]!='RELABEL')
    [(('DELETE', (2, 3), 'PER'), ['O', 'O', 'O', 'O', 'O', 'O'], 2), 
     (('WIDEN1LEFT', 2, 'PER'), ['O', 'B-PER', 'I-PER', 'I-PER', 'O', 'O'], 1), 
     (('SPLIT', 3, 'PER'), ['O', 'O', 'B-PER', 'B-PER', 'O', 'O'], 1), 
     (('NARROW1RIGHT', 3, 'PER'), ['O', 'O', 'B-PER', 'O', 'O', 'O'], 1), 
     (('NARROW1LEFT', 2, 'PER'), ['O', 'O', 'O', 'B-PER', 'O', 'O'], 1), 
     (('WIDEN1RIGHT', 3, 'PER'), ['O', 'O', 'B-PER', 'I-PER', 'I-PER', 'O'], 1)]
    >>> list(e for e in legal_edits([(None, ['O', 'O', 'B-PER', 'B-PER', 'O', 'O'])], ['B-PER']+['I-PER']*5) if e[0][0]!='RELABEL')
    [(('DELETE', (2,), 'PER'), ['O', 'O', 'O', 'B-PER', 'O', 'O'], 2), 
     (('DELETE', (3,), 'PER'), ['O', 'O', 'B-PER', 'O', 'O', 'O'], 2), 
     (('WIDEN1LEFT', 2, 'PER'), ['O', 'B-PER', 'I-PER', 'B-PER', 'O', 'O'], 1), 
     (('MERGE', 3, 'PER'), ['O', 'O', 'B-PER', 'I-PER', 'O', 'O'], 1), 
     (('WIDEN1RIGHT', 3, 'PER'), ['O', 'O', 'B-PER', 'B-PER', 'I-PER', 'O'], 1)]
    >>> list(e for e in legal_edits([(None, ['O', 'O', 'B-PER', 'B-LOC', 'O', 'O'])], ['O']*6) if e[0][0]!='RELABEL')
    [(('DELETE', (2,), 'PER'), ['O', 'O', 'O', 'B-LOC', 'O', 'O'], 2), 
     (('DELETE', (3,), 'LOC'), ['O', 'O', 'B-PER', 'O', 'O', 'O'], 2), 
     (('WIDEN1LEFT', 2, 'PER'), ['O', 'B-PER', 'I-PER', 'B-LOC', 'O', 'O'], 1), 
     (('WIDEN1RIGHT', 3, 'LOC'), ['O', 'O', 'B-PER', 'B-LOC', 'I-LOC', 'O'], 1)]
    >>> list(e for e in legal_edits([(None, ['B-PER', 'I-PER', 'I-PER', 'B-PER'])], ['O']*4) if e[0][0]!='RELABEL')
    [(('DELETE', (0, 1, 2), 'PER'), ['O', 'O', 'O', 'B-PER'], 2), 
     (('DELETE', (3,), 'PER'), ['B-PER', 'I-PER', 'I-PER', 'O'], 2), 
     (('NARROW1LEFT', 0, 'PER'), ['O', 'B-PER', 'I-PER', 'B-PER'], 1), 
     (('NARROW1RIGHT', 2, 'PER'), ['B-PER', 'I-PER', 'O', 'B-PER'], 1)]
    '''
    from_tags = script[-1][1]
    prevStructEdits = [e[:2] for e,r in script[1:]]
    prevEditOps = [e[0].replace('1LEFT','').replace('1RIGHT','') for e in prevStructEdits]
    prevEditOps = [op for k,op in enumerate(prevEditOps) if k==0 or prevEditOps[k-1]!=op]   # remove adjacent duplicates
    # {DELETE}* → {NARROW}* → {WIDEN}* → {SPLIT, RELABEL}* → {MERGE, RELABEL}* → {NARROW at any SPLIT points}* → {INSERT}*
    
    for chk in chunks(from_tags):
        curlbl = lbl(from_tags[chk[0]]) # TODO: what if labels are optional?
        
        # RELABEL
        #if set(chunks(from_tags))==set(chunks(out_tags)): # only consider relabeling after structure is right
        if 'INSERT' not in prevEditOps and not (len(prevEditOps)>=2 and prevEditOps[-1]=='NARROW' and 'SPLIT' in prevEditOps):
            for newlbl in LABELS:
                if newlbl!=curlbl and any(label(out_tags[i])==newlbl for i in chk):
                    result = from_tags[:]
                    for i in chk:
                        result[i] = attachLabel(result[i][0], newlbl)
                    yield ('RELABEL', chk, curlbl, newlbl), result, C_REL
                
        # DELETE
        if not prevEditOps or prevEditOps[0]=='DELETE': # any DELETEs must come first
            result = from_tags[:]
            prevI = None
            for i in chk:
                if prevI is not None and prevI+1<i: # we are after a gap
                    for j in range(prevI+1,i):
                        result[j] = toOutOfGap(result[j])
                result[i] = toOutside(result[i])     # TODO: what if the chunk has a gap? need to uppercase the contents
                prevI = i
            yield ('DELETE', chk, curlbl), result, C_DEL
        
    # INSERT
    for chk in chunks(out_tags):    # only consider inserting chunks that are in the output
        if all(isOutside(from_tags[i]) for i in chk) and (all(isInGap(from_tags[i]) for i in chk) or all(not isInGap(from_tags[i]) for i in chk)):   # there is room to insert the chunk
            newlbl = lbl(out_tags[chk[0]])
            result = from_tags[:]
            result[chk[0]] = attachLabel(('B' if not isInGap(result[chk[0]]) else 'b'), newlbl)
            prevI = chk[0]
            for i in chk[1:]:
                if prevI+1<i: # we are after a gap
                    for j in range(prevI+1,i):
                        result[j] = toInGap(result[j])
                result[i] = attachLabel(('I' if not isInGap(result[i]) else 'i'), newlbl)
                prevI = i
            yield ('INSERT', chk, newlbl), result, C_INS
            
    # NARROW, WIDEN, SPLIT, MERGE
    if len(prevEditOps)<1 or prevEditOps[-1]!='INSERT':
        for (i,a),b in zip(enumerate(from_tags[:-1]),from_tags[1:]):
            if isInGap(a)!=isInGap(b): continue # there will be separate operations to change gaps
            elif isOutside(a) and isBegin(b): # WIDEN1LEFT
                if set(prevEditOps) <= {'DELETE','NARROW','WIDEN'}:
                    if ('WIDEN1LEFT', i+1) not in prevStructEdits and ('NARROW1LEFT', i) not in prevStructEdits:
                        #and not isOutside(out_tags[i]):  # expediency
                        result = from_tags[:]
                        result[i] = result[i+1]
                        result[i+1] = toInside(result[i+1])
                        yield ('WIDEN1LEFT', i+1, lbl(b)), result, C_WID
            elif not isOutside(a) and isOutside(b): # WIDEN1RIGHT
                if set(prevEditOps) <= {'DELETE','NARROW','WIDEN'}:
                    if ('WIDEN1RIGHT', i) not in prevStructEdits and ('NARROW1RIGHT', i+1) not in prevStructEdits:
                        #and not isOutside(out_tags[i+1]):    # expediency
                        result = from_tags[:]
                        result[i+1] = toInside(result[i])
                        yield ('WIDEN1RIGHT', i, lbl(a)), result, C_WID
            elif not isOutside(a) and not isOutside(b) and lbl(a)==lbl(b):  # SPLIT/MERGE/NARROW
                if isBegin(b): # different chunks--MERGE
                    if not (len(prevEditOps)>=2 and prevEditOps[-1]=='NARROW' and 'SPLIT' in prevEditOps):
                        if isInGap(out_tags[i])==isInGap(out_tags[i+1]) and isInside(out_tags[i+1]):   # expediency constraint (opp. of SPLIT)
                            result = from_tags[:]
                            result[i+1] = toInside(result[i+1])
                            yield ('MERGE', i+1, lbl(a)), result, C_MRG
                else:   # same chunk
                    # SPLIT
                    if 'MERGE' not in prevEditOps and not (len(prevEditOps)>=2 and prevEditOps[-1]=='NARROW' and 'SPLIT' in prevEditOps):
                        if not (isOutside(out_tags[i]) and isOutside(out_tags[i+1])) \
                           and not (isInGap(out_tags[i])==isInGap(out_tags[i+1]) and isInside(out_tags[i+1])): # expediency constraint: split parts must belong to different chunks of the output (at most one may belong to no chunk)
                            result = from_tags[:]
                            result[i+1] = toBegin(result[i+1])
                            thecost = C_SPL
                            yield ('SPLIT', i+1, lbl(a)), result, thecost
                    # NARROW
                    if not ({'MERGE','WIDEN','RELABEL'} & set(prevEditOps) and 'SPLIT' not in prevEditOps):
                        if i+2==len(from_tags) or not (isInside(from_tags[i+2]) or isInGap(from_tags[i+2])):    # need to make sure 'b' is not in the middle of a chunk
                            # (it doesn't matter what 'a' is so long as it's not the last token in a gap)
                            if ('NARROW1RIGHT', i+1) not in prevStructEdits and ('WIDEN1RIGHT', i) not in prevStructEdits:
                                result = from_tags[:]
                                result[i+1] = toOutside(result[i+1])
                                yield ('NARROW1RIGHT', i+1, lbl(a)), result, C_NAR
                        if isBegin(a):
                            if ('NARROW1LEFT', i) not in prevStructEdits and ('WIDEN1LEFT', i+1) not in prevStructEdits:
                                result = from_tags[:]
                                result[i] = toOutside(result[i])
                                result[i+1] = toBegin(result[i+1])
                                yield ('NARROW1LEFT', i, lbl(a)), result, C_NAR

def firstint(x):
    if isinstance(x,int): return x
    return x[0]

def best_script(in_tags, out_tags):
    '''
    Find a minimum-cost edit script (subject to constraints) that transforms 
    'in_tags' (the source) into 'out_tags' (the target) via the edit operations defined 
    in legal_edits().
    
    This function runs A* search subject to the described constraints on search moves. 
    The A* heuristic is explained in the code. There is also a tiebreaker heuristic 
    (for candidates with equal A* estimates) that prefers candidates whose output 
    has more tags in common with the target tagging. For the Torture Test examples below, 
    we find that the A* heuristic and the tiebreaker heuristic reduce runtime considerably.
    
    Edit scripts (derivations) imply alignments between chunks in the source 
    that are not DELETEd, and chunks in the target that are not INSERTed.
    
    (TODO: Write a function to extract alignments from a script)
    
    >>> LABELS |= set(['PER', 'ORG', 'LOC', 'evt'])
    >>> best_script(['O'], ['O'])
    (0.0, [])
    >>> best_script(['O'], ['B-LOC'])
    (2.0, [(('INSERT', (0,), 'LOC'), ['B-LOC'])])
    >>> best_script(['B-evt'], ['B-LOC'])
    (1.0, [(('RELABEL', (0,), 'evt', 'LOC'), ['B-LOC'])])
    >>> best_script(['B-PER', 'I-PER'], ['B-PER', 'B-PER'])
    (1.0, [(('SPLIT', 1, 'PER'), ['B-PER', 'B-PER'])])
    >>> best_script(['B-PER', 'I-PER'], ['B-PER', 'B-LOC'])
    (2.0, [(('SPLIT', 1, 'PER'), ['B-PER', 'B-PER']), (('RELABEL', (1,), 'PER', 'LOC'), ['B-PER', 'B-LOC'])])
    >>> best_script(['B-evt', 'o', 'I-evt'], ['O', 'B-LOC', 'O'])
    (4.0, [(('DELETE', (0, 2), 'evt'), ['O', 'O', 'O']), (('INSERT', (1,), 'LOC'), ['O', 'B-LOC', 'O'])])
    >>> best_script(['O', 'O', 'O'], ['B-LOC', 'I-LOC', 'O'])
    (2.0, [(('INSERT', (0, 1), 'LOC'), ['B-LOC', 'I-LOC', 'O'])])
    >>> best_script(['O', 'B-PER', 'o', 'I-PER'], ['B-evt', 'o', 'I-evt', 'O'])
    (4.0, [(('DELETE', (1, 3), 'PER'), ['O', 'O', 'O', 'O']), (('INSERT', (0, 2), 'evt'), ['B-evt', 'o', 'I-evt', 'O'])])
    >>> best_script(['O', 'O', 'B-PER', 'O', 'O'], ['O']*5)
    (2.0, [(('DELETE', (2,), 'PER'), ['O', 'O', 'O', 'O', 'O'])])
    >>> best_script(['O', 'O', 'B-PER', 'I-PER', 'O', 'O'], ['O']*6)
    (2.0, [(('DELETE', (2, 3), 'PER'), ['O', 'O', 'O', 'O', 'O', 'O'])])
    >>> best_script(['B-PER', 'I-PER', 'I-PER', 'B-PER'], ['B-PER', 'I-PER', 'B-LOC', 'I-LOC'])
    (3.0, [(('NARROW1RIGHT', 2, 'PER'), ['B-PER', 'I-PER', 'O', 'B-PER']), 
    (('WIDEN1LEFT', 3, 'PER'), ['B-PER', 'I-PER', 'B-PER', 'I-PER']), 
    (('RELABEL', (2, 3), 'PER', 'LOC'), ['B-PER', 'I-PER', 'B-LOC', 'I-LOC'])])
    
    
    Constraints on MERGE and SPLIT
    ==============================
    
    Adjacent chunks can usually be MERGEd, so long as their labels match:
    
    >>> best_script(['O', 'O', 'B-PER', 'B-PER', 'O', 'O'], ['O', 'O', 'B-PER', 'I-PER', 'O', 'O'])
    (1.0, [(('MERGE', 3, 'PER'), ['O', 'O', 'B-PER', 'I-PER', 'O', 'O'])])
    >>> best_script(['O', 'O', 'B-PER', 'B-LOC', 'O', 'O'], ['O', 'O', 'B-PER', 'I-PER', 'O', 'O'])
    (2.0, [(('RELABEL', (3,), 'LOC', 'PER'), ['O', 'O', 'B-PER', 'B-PER', 'O', 'O']), (('MERGE', 3, 'PER'), ['O', 'O', 'B-PER', 'I-PER', 'O', 'O'])])

    But MERGE is not allowed if there is no overlapping chunk on the target side.
    In some cases, this prevents cheaper MERGE, then DELETE solutions. 
    (Intuitively, it is weird to assume that 2 extra chunks are close to being 1 extra chunk.)
    For example:
    
    >>> best_script(['O', 'O', 'B-PER', 'B-PER', 'O', 'O'], ['O']*6)
    (4.0, [(('DELETE', (2,), 'PER'), ['O', 'O', 'O', 'B-PER', 'O', 'O']), (('DELETE', (3,), 'PER'), ['O', 'O', 'O', 'O', 'O', 'O'])])
    >>> best_script(['B-PER', 'I-PER', 'I-PER', 'B-PER'], ['O']*4)
    (4.0, [(('DELETE', (0, 1, 2), 'PER'), ['O', 'O', 'O', 'B-PER']), (('DELETE', (3,), 'PER'), ['O', 'O', 'O', 'O'])])
    
    A parallel constraint applies to SPLIT:
    
    >>> best_script(['O', 'O', 'B-PER', 'I-PER', 'O', 'O'], ['O', 'O', 'B-PER', 'B-PER', 'O', 'O'])
    (1.0, [(('SPLIT', 3, 'PER'), ['O', 'O', 'B-PER', 'B-PER', 'O', 'O'])])
    >>> best_script(['O']*6, ['O', 'O', 'B-PER', 'B-PER', 'O', 'O'])
    (4.0, [(('INSERT', (2,), 'PER'), ['O', 'O', 'B-PER', 'O', 'O', 'O']), (('INSERT', (3,), 'PER'), ['O', 'O', 'B-PER', 'B-PER', 'O', 'O'])])
    
    (Otherwise the above would be INSERT, followed by SPLIT, for a cost of 3.0.)
    
    
    Constraining Search Sequences
    =============================
    
    For more complicated examples (see Torture Tests below), we need to be careful about 
    the search space--otherwise search becomes extremely slow. One tactic we use is 
    to reduce some of the spurious ambiguity in the ordering of edits: 
    (a) many edits affect nonoverlapping regions and thus do not interact, so we require them 
    to be basically left-to-right; and (b) orders of certain kinds edit operations can
    be eliminated as unnecessary without eliminating a preferred solution. We require 
    edits to proceed in 7 stages:
    
    {DELETE}* → {NARROW}* → {WIDEN}* → {SPLIT, RELABEL}* → {MERGE, RELABEL}* → {NARROW at any SPLIT points}* → {INSERT}*
    
    This, for instance, eliminates the possibility of WIDEN before NARROW 
    (which is never required, but NARROW is sometimes required to create the necessary 
    preconditions for WIDEN). It also eliminates a couple of lowest-cost edit sequences 
    on the grounds that they are unintuitive. For two adjacent chunks with the same label, 
    MERGE followed by DELETE will be lower cost than two DELETEs, but it gives a misleading 
    impression about the number of source-side chunks that are not at all represented 
    on the target side.
    
    To ensure edits are roughly left-to-right: within each stage, we do not consider edits 
    that are way earlier in the sequence than the previous edit. (Specifically, 
    the first token position in the edit should not be more than 2 tokens to the left of 
    the first token position in the previous edit in the script. TODO: be more specific.
    See `prevEditI` in the code.) This allows some leeway to allow for edits facilitated 
    by an edit to their right:
    
    >>> best_script(['O', 'O', 'O', 'B-PER'], ['B-PER', 'I-PER', 'I-PER', 'I-PER'])
    (3.0, [(('WIDEN1LEFT', 3, 'PER'), ['O', 'O', 'B-PER', 'I-PER']), 
    (('WIDEN1LEFT', 2, 'PER'), ['O', 'B-PER', 'I-PER', 'I-PER']), 
    (('WIDEN1LEFT', 1, 'PER'), ['B-PER', 'I-PER', 'I-PER', 'I-PER'])])
    
    We also avoid searching over redundant edits:
     - Edits that are identical to a previous edit in the script are not considered.
     - With WIDEN/NARROW, edits that reverse a previous edit in the script are not considered: 
       e.g., WIDEN1RIGHT @ 5, and later NARROW1RIGHT @ 6 (or vice versa).
    
    
    Justification for the 7-Stage Ordering
    ======================================
    
    Rationale for NARROW < WIDEN: B-PER I-PER B-LOC → B-PER B-LOC I-LOC
    (But WIDEN < NARROW is never necessary, absent an intervening SPLIT?)
    
    Rationale for WIDEN < MERGE: B-PER O B-PER → B-PER I-PER I-PER
    Rationale for SPLIT < NARROW: B-PER I-PER I-PER → B-PER O B-PER
    
    Rationale for RELABEL < SPLIT: B-PER I-PER I-PER → B-LOC B-LOC I-LOC
    Rationale for MERGE < RELABEL: B-LOC B-LOC I-LOC → B-PER I-PER I-PER
    
    Rationale for RELABEL < MERGE: B-PER B-LOC I-LOC → B-PER I-PER I-PER
    Rationale for SPLIT < RELABEL: B-PER I-PER I-PER → B-PER B-LOC I-LOC
    
    Rationale for SPLIT < {MERGE,RELABEL}: B-PER B-PER I-PER I-PER → B-PER I-PER I-PER B-LOC without causing everything to align to everything
    Rationale for {SPLIT,RELABEL} < MERGE: B-PER I-PER I-PER B-LOC → B-PER B-PER I-PER I-PER without causing everything to align to everything
    
    MERGE < RELABEL < SPLIT sometimes produces a lower cost than placing SPLITs before 
    merges, but this has the disadvantage of imprecise alignments (all token positions 
    involved in the MERGE will be aligned). An example is 
      B-PER B-PER I-PER B-PER → B-LOC I-LOC B-LOC I-LOC 
    The derivation MERGE → MERGE → RELABEL → SPLIT would result in everything being aligned 
    to everything; we prefer a solution that gives better alignments but costs more, like

    >>> best_script(['B-PER', 'B-PER', 'I-PER', 'B-PER'], ['B-LOC', 'I-LOC', 'B-LOC', 'I-LOC'])
    (5.0, [(('NARROW1LEFT', 1, 'PER'), ['B-PER', 'O', 'B-PER', 'B-PER']), 
    (('WIDEN1RIGHT', 0, 'PER'), ['B-PER', 'I-PER', 'B-PER', 'B-PER']), 
    (('RELABEL', (0, 1), 'PER', 'LOC'), ['B-LOC', 'I-LOC', 'B-PER', 'B-PER']), 
    (('MERGE', 3, 'PER'), ['B-LOC', 'I-LOC', 'B-PER', 'I-PER']), 
    (('RELABEL', (2, 3), 'PER', 'LOC'), ['B-LOC', 'I-LOC', 'B-LOC', 'I-LOC'])])
    
    (SPLIT → MERGE → MERGE → RELABEL → RELABEL also has cost 5.0.)
    
    
    Torture Tests
    =============
    
    >>> best_script( \
    ['O',     'B-evt', 'o', 'b-PER', 'I-evt', 'I-evt', 'B-PER', 'O', 'B-ORG', 'I-ORG', 'B-ORG', 'O',     'O',     'O'], \
    ['B-evt', 'I-evt', 'o', 'b-PER', 'I-evt', 'B-PER', 'I-PER', 'O', 'B-LOC', 'I-LOC', 'B-ORG', 'B-evt', 'b-MIS', 'I-evt'])
    (8.0, [(('NARROW1RIGHT', 5, 'evt'), ['O', 'B-evt', 'o', 'b-PER', 'I-evt', 'O', 'B-PER', 'O', 'B-ORG', 'I-ORG', 'B-ORG', 'O', 'O', 'O']), 
    (('WIDEN1LEFT', 1, 'evt'), ['B-evt', 'I-evt', 'o', 'b-PER', 'I-evt', 'O', 'B-PER', 'O', 'B-ORG', 'I-ORG', 'B-ORG', 'O', 'O', 'O']),
    (('WIDEN1LEFT', 6, 'PER'), ['B-evt', 'I-evt', 'o', 'b-PER', 'I-evt', 'B-PER', 'I-PER', 'O', 'B-ORG', 'I-ORG', 'B-ORG', 'O', 'O', 'O']), 
    (('RELABEL', (8, 9), 'ORG', 'LOC'), ['B-evt', 'I-evt', 'o', 'b-PER', 'I-evt', 'B-PER', 'I-PER', 'O', 'B-LOC', 'I-LOC', 'B-ORG', 'O', 'O', 'O']), 
    (('INSERT', (11, 13), 'evt'), ['B-evt', 'I-evt', 'o', 'b-PER', 'I-evt', 'B-PER', 'I-PER', 'O', 'B-LOC', 'I-LOC', 'B-ORG', 'B-evt', 'o', 'I-evt']), 
    (('INSERT', (12,), 'MIS'), ['B-evt', 'I-evt', 'o', 'b-PER', 'I-evt', 'B-PER', 'I-PER', 'O', 'B-LOC', 'I-LOC', 'B-ORG', 'B-evt', 'b-MIS', 'I-evt'])])
    >>> best_script( \
    ['O',     'B-evt', 'O',     'B-PER', 'B-evt', 'I-evt', 'B-LOC', 'I-LOC', 'B-PER', 'I-PER'], \
    ['B-evt', 'I-evt', 'B-PER', 'I-PER', 'O',     'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'I-ORG'])
    (9.0, [(('NARROW1LEFT', 4, 'evt'), ['O', 'B-evt', 'O', 'B-PER', 'O', 'B-evt', 'B-LOC', 'I-LOC', 'B-PER', 'I-PER']), 
    (('NARROW1LEFT', 6, 'LOC'), ['O', 'B-evt', 'O', 'B-PER', 'O', 'B-evt', 'O', 'B-LOC', 'B-PER', 'I-PER']), 
    (('WIDEN1LEFT', 1, 'evt'), ['B-evt', 'I-evt', 'O', 'B-PER', 'O', 'B-evt', 'O', 'B-LOC', 'B-PER', 'I-PER']), 
    (('WIDEN1LEFT', 3, 'PER'), ['B-evt', 'I-evt', 'B-PER', 'I-PER', 'O', 'B-evt', 'O', 'B-LOC', 'B-PER', 'I-PER']), 
    (('WIDEN1RIGHT', 5, 'evt'), ['B-evt', 'I-evt', 'B-PER', 'I-PER', 'O', 'B-evt', 'I-evt', 'B-LOC', 'B-PER', 'I-PER']), 
    (('RELABEL', (5, 6), 'evt', 'PER'), ['B-evt', 'I-evt', 'B-PER', 'I-PER', 'O', 'B-PER', 'I-PER', 'B-LOC', 'B-PER', 'I-PER']), 
    (('RELABEL', (7,), 'LOC', 'ORG'), ['B-evt', 'I-evt', 'B-PER', 'I-PER', 'O', 'B-PER', 'I-PER', 'B-ORG', 'B-PER', 'I-PER']), 
    (('RELABEL', (8, 9), 'PER', 'ORG'), ['B-evt', 'I-evt', 'B-PER', 'I-PER', 'O', 'B-PER', 'I-PER', 'B-ORG', 'B-ORG', 'I-ORG']), 
    (('MERGE', 8, 'ORG'), ['B-evt', 'I-evt', 'B-PER', 'I-PER', 'O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'I-ORG'])])
    >>> best_script( \
    ['O',     'B-evt', 'o',     'b-PER', 'I-evt', 'I-evt', 'B-LOC', 'I-LOC', 'B-PER', 'I-PER'], \
    ['B-evt', 'I-evt', 'B-PER', 'I-PER', 'O',     'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'I-ORG']) # has gap in source only
    (10.0, [(('DELETE', (1, 4, 5), 'evt'), ['O', 'O', 'O', 'B-PER', 'O', 'O', 'B-LOC', 'I-LOC', 'B-PER', 'I-PER']), 
    (('NARROW1RIGHT', 7, 'LOC'), ['O', 'O', 'O', 'B-PER', 'O', 'O', 'B-LOC', 'O', 'B-PER', 'I-PER']), 
    (('WIDEN1LEFT', 3, 'PER'), ['O', 'O', 'B-PER', 'I-PER', 'O', 'O', 'B-LOC', 'O', 'B-PER', 'I-PER']), 
    (('WIDEN1LEFT', 6, 'LOC'), ['O', 'O', 'B-PER', 'I-PER', 'O', 'B-LOC', 'I-LOC', 'O', 'B-PER', 'I-PER']), 
    (('WIDEN1LEFT', 8, 'PER'), ['O', 'O', 'B-PER', 'I-PER', 'O', 'B-LOC', 'I-LOC', 'B-PER', 'I-PER', 'I-PER']), 
    (('RELABEL', (5, 6), 'LOC', 'PER'), ['O', 'O', 'B-PER', 'I-PER', 'O', 'B-PER', 'I-PER', 'B-PER', 'I-PER', 'I-PER']), 
    (('RELABEL', (7, 8, 9), 'PER', 'ORG'), ['O', 'O', 'B-PER', 'I-PER', 'O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'I-ORG']), 
    (('INSERT', (0, 1), 'evt'), ['B-evt', 'I-evt', 'B-PER', 'I-PER', 'O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'I-ORG'])])
    '''
    assert isValidTagging(in_tags)
    assert isValidTagging(out_tags)
    assert len(in_tags)==len(out_tags),'This formulation of edits is over tags, so it requires sequences of equal length'
    
    q = []  # priority queue, accessed by heapq.heappush(heap, cost) heapq.heappop(q)
    heapq.heappush(q, ((0.0, 0, 0.0), [(None, list(in_tags))])) # edit script encoding only the initial state

    best_script = best_script_cost = None

    while q:
        (hcost,dcost,cost),script = heapq.heappop(q) # sequence of (edit, result) pairs. script[-1][1] is the tag sequence after all edits in the script
        if len(script)>1:
            prevEdits = zip(*(script[1:]))[0]
            prevEditI = firstint(prevEdits[-1][1])
        else:
            prevEdits = []
            prevEditI = 0
            
        if script and script[-1][1]==out_tags:
            best_script = script
            best_script_cost = cost
            break
        for edit,result,editcost in legal_edits(script, out_tags):
            if edit in prevEdits or (prevEditI-2>firstint(edit[1]) and edit[0][:3]==prevEdits[-1][0][:3]):
                #print(edit, script, file=sys.stderr)
                continue # don't repeat a previous edit in this script, and require edits to be mostly left-to-right
            #print(hcost, dcost, cost, zip(*script)[0], edit, file=sys.stderr)
            assert len(result)==len(in_tags)
            assert isValidTagging(result),(script,edit,result)
            #heuristic = abs(sum(1 for t in result if isBegin(t))-sum(1 for t in out_tags if isBegin(t))) # TODO: verify this is admissible!
            #heuristic = sum(1 for s,t in zip(result,out_tags) if not isInside(s) and not isInside(t) and (isBegin(s)!=isBegin(t)))
            nDifferences = sum(1 for s,t in zip(result,out_tags) if s!=t)  # not optimal! but a good tiebreaker?
            #nBDifferences = sum(1 for s,t in zip(result,out_tags) if s!=t and (isBegin(s) or isBegin(t)) and not (isInside(s) or isInside(t)))  # ??
            #heuristic = nDifferences - abs(sum(1 for t in result if isInside(t))-sum(1 for t in out_tags if isInside(t)))
            unmatchedSrcChunks = sum(1 for chk in chunks(result) if any(result[i]!=out_tags[i] for i in chk))
            unmatchedTgtChunks = sum(1 for chk in chunks(out_tags) if any(result[i]!=out_tags[i] for i in chk))
            adjacentSrcChunksSameLabel = sum(1 for i,t in enumerate(result) if i>0 and isBegin(t) and label(t)==label(result[i-1]) and isInGap(t)==isInGap(result[i-1]))
            adjacentTgtChunksSameLabel = sum(1 for i,t in enumerate(out_tags) if i>0 and isBegin(t) and label(t)==label(out_tags[i-1]) and isInGap(t)==isInGap(out_tags[i-1]))
            adjacentSrcChunksSameLabel = adjacentTgtChunksSameLabel = 0 # TEST. makes the heuristic non-admissible!
            oMatchedSrcChunks = sum(1 for chk in chunks(result) if all(isOutside(out_tags[i]) for i in chk)) # number of chunks in 'result' matching all O's in the output
            oMatchedTgtChunks = sum(1 for chk in chunks(out_tags) if all(isOutside(result[i]) for i in chk))
            
            #boundaryMismatchChunks = len(set(chunks(result)) ^ set(chunks(out_tags)))
            #boundaryMatchLabelMismatchChunks = sum(1 for chk in (set(chunks(result)) & set(chunks(out_tags))) if label(result[chk[0]])!=label(out_tags[chk[0]]))
            #heuristic = boundaryMatchLabelMismatchChunks+.5*boundaryMismatchChunks
            
            '''
            A* heuristic:
            The following is *almost* a lower bound on the edit cost: 
            
               number of target chunks that are not exactly matched in the source
                                               +
               number of source chunks that do not overlap with any target chunks
               
            This also applies to the reverse of source/target, so we take the max of the 
            two directions for a tighter bound.
            
            However, this is not *quite* a lower bound because of SPLIT/MERGE: 
            e.g. B-LOC I-LOC → B-LOC B-LOC has 2 unmatched chunks on target side but 
            only 1.0 cost is incurred with the MERGE operation. To account for this, 
            we subtract the number of Begin tags immediately following a chunk with 
            the same label (which could be MERGEd without additional edits). 
            This makes the heuristic admissible.

            (We do not have to correct the other, "O"-matching, term because B-LOC B-LOC → O O 
            is required to be two DELETEs, not a MERGE followed by a DELETE.)            
            '''
            heuristic = max(unmatchedTgtChunks - adjacentTgtChunksSameLabel + oMatchedSrcChunks, 
                            unmatchedSrcChunks - adjacentSrcChunksSameLabel + oMatchedTgtChunks)
            
            '''
            Tiebreaker heuristic: total number of source/target chunks not exactly matched 
            on the other side. Lower is better. The intuition is that it rewards partial 
            derivations that haved "locked in" more fully correct chunks. 
            
            (This would not be admissible as an A* heuristic; it is only used when 
            the A* estimate has multiple best scoring candidates.)
            '''
            nUnmatched = (unmatchedSrcChunks+unmatchedTgtChunks)
            #heapq.heappush(q, ((cost+editcost+heuristic,cost+editcost), script[:]+[(edit,result)]))
            
            '''
            Number of MERGE edits in the script that are followed later by some SPLIT, 
            possibly in the current edit.
            (Conflates all MERGEs and all SPLITs, which is slightly hacky: 
            can probably do better by looking at indices)
            '''
            #anteSplitMerges = sum(1 for i,e in enumerate(prevEdits) if e[0]=='MERGE' and (edit[0]=='SPLIT' or any(e2[0]=='SPLIT' for e2 in prevEdits[i+1:])))
            
            heapq.heappush(q, ((cost+editcost+heuristic,nUnmatched,cost+editcost), script[:]+[(edit,result)]))
            
    return best_script_cost, best_script[1:]    # don't include the initial state in the returned script

def format_script(in_tags, script, tot_cost):
    firstline = ' '*42 + ' '.join('{:6}'.format(tag) for tag in in_tags) + '\n'
    return firstline + '\n'.join('  {op:12} {args:26} {result}'.format(op=edit[0], 
        args=edit[1:], result=' '.join('{:6}'.format(tag) for tag in result)) for edit,result in script)+'\n___\n{}'.format(tot_cost)

def Fscore(prec, rec):
    if prec==0.0 or rec==0.0: return float('nan')
    return 2*prec*rec/(prec+rec)

def score(gold_tags, pred_tags):
    assert len(gold_tags)==len(pred_tags)
    cost, script = best_script(pred_tags, gold_tags)
    print('Pred -> Gold edit script:')
    print(format_script(pred_tags, script,cost))
    print()
    pcost, pscript = best_script(pred_tags, ['O']*len(gold_tags))
    rcost, rscript = best_script(['O']*len(pred_tags), gold_tags)
    print('DELETE all pred, INSERT all gold edit script:')
    print(format_script(pred_tags,pscript+rscript,pcost+rcost))
    print()
    
    print('cost =',cost)
    print('DELETE all pred cost =',pcost)
    print('INSERT all gold cost =',rcost)
    overall_score = 1 - cost/(pcost+rcost)
    print('Score = 1 - {}/{} = {}'.format(cost,(pcost+rcost),overall_score))
    #f = Fscore(cost/pdenom, cost/rdenom)
    #print('recall denom =',rdenom)
    #print('precision denom =',pdenom)
    #print('P = ',cost/pdenom, 'R = ',cost/rdenom, 'F = ',f)
    return overall_score

def load(goldAndPredF):
    data = []
    
    c = Counter()
    pp = []
    gg = []
    for ln in goldAndPredF:
        if (not ln.strip()) and gg:  # sequence break
            assert isValidTagging(gg)
            assert isValidTagging(pp)
            data.append((gg,pp))
            pp = []
            gg = []
            continue
        w,g,p = ln.strip().split('\t')
        lg,lp = label(g),label(p)
        if lg and lg not in LABELS: LABELS.append(lg)
        if lp and lp not in LABELS: LABELS.append(lp)
        c['NTags'] += 1
        if g==p:
            c['NCorrectTags'] += 1
    if gg:
        assert isValidTagging(gg)
        assert isValidTagging(pp)
        data.append((gg,pp))
    print(c)
    
    return data
    
def main():
    data = load(sys.stdin)
    for gg,pp in data:
        best_script(tuple(gg), tuple(pp))

if __name__=='__main__':
    # running all the doctests takes 10s on my machine
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
    print('BASE COSTS: INS={} | DEL={} | WID={} | NAR={} | SPL={} | MRG={} | REL={}'.format(C_INS,C_DEL,C_WID,C_NAR,C_SPL,C_MRG,C_REL))
    print()
    print('EXAMPLE:')
    pp = ['O',     'B-evt', 'O',     'B-PER', 'B-evt', 'I-evt', 'B-LOC', 'I-LOC', 'B-PER', 'I-PER']
    gg = ['B-evt', 'I-evt', 'B-PER', 'I-PER', 'O',     'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'I-ORG']
    print('Pred:', pp)
    print('Gold:', gg)
    score(gg, pp)
