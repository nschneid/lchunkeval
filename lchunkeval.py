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
        #        0    1        2    3        4    5    6        7    8        9
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
    RELABEL: B-PER (I-PER)* ↔ B-ORG (I-ORG)*
      on the condition that at least one of the tokens in the chunk is labeled with ORG in the output
    INSERT(DELETE): O (O)* → B-PER (I-PER)*
      INSERT on the condition that the inserted chunk is in the output
    SPLIT(MERGE): B-PER I-PER → B-PER B-PER; I-PER I-PER → I-PER B-PER
    NARROW1LEFT(WIDEN1LEFT): B-PER I-PER → O B-PER
    NARROW1RIGHT: B|I-PER I-PER ] → O  (where "]" means "followed by a non-I tag or the end of the sequence")
    WIDEN1RIGHT: B-PER O → B-PER I-PER; I-PER O → I-PER I-PER
    
    The above rules also work with gappy expressions, so long as a rule application 
    does not affect tags both inside and outside of a gap.
    
    #(INSERT-GAP, DELETE-GAP, WIDEN-GAP-{LEFT,RIGHT}, NARROW-GAP-{LEFT,RIGHT}, STRENGTHEN, WEAKEN)
    
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
    
    for chk in chunks(from_tags):
        curlbl = lbl(from_tags[chk[0]]) # TODO: what if labels are optional?
        
        # RELABEL
        if set(chunks(from_tags))==set(chunks(out_tags)): # only consider relabeling after structure is right
        #if True:
            for newlbl in LABELS:
                if newlbl!=curlbl and any(label(out_tags[i])==newlbl for i in chk):
                    result = from_tags[:]
                    for i in chk:
                        result[i] = attachLabel(result[i][0], newlbl)
                    yield ('RELABEL', chk, curlbl, newlbl), result, C_REL
                
        # DELETE
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
            
    for (i,a),b in zip(enumerate(from_tags[:-1]),from_tags[1:]):
        if isInGap(a)!=isInGap(b): continue # there will be separate operations to change gaps
        elif isOutside(a) and isBegin(b): # WIDEN1LEFT
            if ('WIDEN1LEFT', i+1) not in prevStructEdits and ('NARROW1LEFT', i) not in prevStructEdits:
                #and not isOutside(out_tags[i]):  # expediency
                result = from_tags[:]
                result[i] = result[i+1]
                result[i+1] = toInside(result[i+1])
                yield ('WIDEN1LEFT', i+1, lbl(b)), result, C_WID
        elif not isOutside(a) and isOutside(b): # WIDEN1RIGHT
            if ('WIDEN1RIGHT', i) not in prevStructEdits and ('NARROW1RIGHT', i+1) not in prevStructEdits:
                #and not isOutside(out_tags[i+1]):    # expediency
                result = from_tags[:]
                result[i+1] = toInside(result[i])
                yield ('WIDEN1RIGHT', i, lbl(a)), result, C_WID
        elif not isOutside(a) and not isOutside(b) and lbl(a)==lbl(b):  # SPLIT/MERGE/NARROW
            if isBegin(b): # different chunks--MERGE
                if isInGap(out_tags[i])==isInGap(out_tags[i+1]) and isInside(out_tags[i+1]):   # expediency constraint (opp. of SPLIT)
                    result = from_tags[:]
                    result[i+1] = toInside(result[i+1])
                    yield ('MERGE', i+1, lbl(a)), result, C_MRG
            else:   # same chunk
                # SPLIT
                if not (isOutside(out_tags[i]) and isOutside(out_tags[i+1])) \
                   and not (isInGap(out_tags[i])==isInGap(out_tags[i+1]) and isInside(out_tags[i+1])): # expediency constraint: split parts must belong to different chunks of the output (at most one may belong to no chunk)
                    result = from_tags[:]
                    result[i+1] = toBegin(result[i+1])
                    thecost = C_SPL
                    yield ('SPLIT', i+1, lbl(a)), result, thecost
                # NARROW
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
    Find a minimum-cost edit script that transforms 'in_tags' (the source) into 'out_tags' 
    (the target) via the edit operations defined in legal_edits().
    
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
    
    Adjacent chunks can usually be MERGEd, so long as their labels match:
    
    >>> best_script(['O', 'O', 'B-PER', 'B-PER', 'O', 'O'], ['O', 'O', 'B-PER', 'I-PER', 'O', 'O'])
    (1.0, [(('MERGE', 3, 'PER'), ['O', 'O', 'B-PER', 'I-PER', 'O', 'O'])])
    >>> best_script(['O', 'O', 'B-PER', 'B-LOC', 'O', 'O'], ['O', 'O', 'B-PER', 'I-PER', 'O', 'O'])
    (3.0, [(('DELETE', (3,), 'LOC'), ['O', 'O', 'B-PER', 'O', 'O', 'O']), (('WIDEN1RIGHT', 2, 'PER'), ['O', 'O', 'B-PER', 'I-PER', 'O', 'O'])])

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
    
    Edit scripts (derivations) imply alignments between chunks in the source 
    that are not DELETEd, and chunks in the target that are not INSERTed.
    A complication that can arise when MERGEs and SPLITs are used in combination
    is that multiple lowest-cost solutions can give different alignments:
    
    >>> best_script(['B-PER', 'I-PER', 'I-PER', 'B-PER'], ['B-PER', 'I-PER', 'B-LOC', 'I-LOC'])    \
    # multiple lowest-cost solutions, resulting in different alignments. \
    # the minimize-ante-split-merges tiebreaking strategy helps avoid crossing alignments.
    (3.0, [(('NARROW1RIGHT', 2, 'PER'), ['B-PER', 'I-PER', 'O', 'B-PER']), 
    (('WIDEN1LEFT', 3, 'PER'), ['B-PER', 'I-PER', 'B-PER', 'I-PER']), 
    (('RELABEL', (2, 3), 'PER', 'LOC'), ['B-PER', 'I-PER', 'B-LOC', 'I-LOC'])])
    
    This derivation is preferred over another with equal cost, 
    which first MERGEs everything into a single chunk and then splits it in half, 
    because the latter would mean that all of the final chunks are aligned 
    to all of the original chunks (so there will be crossing alignments). 
    To prefer the derivation with the better alignment, we use a heuristic tiebreaker 
    between equal (heuristic-)cost search steps: the tiebreaker is to choose the script
    having the fewest MERGE edits atecedent to some SPLIT edit.
    
    The complicated examples below show that we need to be careful about the search space;
    otherwise search becomes extremely slow. Tactics include:
    
    a. Only consider RELABEL operations at the end, after the structure matches.
       And only consider the label seen in the target--so these operations are trivial. 
       (TODO: Reconsider, because it will sometimes cause extra ops: 
       B-PER I-PER I-PER → B-LOC B-LOC I-LOC; B-PER B-LOC I-LOC → B-PER I-PER I-PER. 
       Maybe allow just SPLIT/MERGE after RELABEL?)
    b. Ensure edits are roughly left-to-right: block edits that are way earlier in the 
       sequence than the previous edit. (Specifically, the first token position 
       in the edit should not be more than 2 tokens to the left of 
       the first token position in the previous edit in the script. TODO: be more specific than "in the edit".
       Implemented separately for RELABEL operations in light of (a). See `prevEditI` in the code.)
       This eliminates some spurious ambiguity in the order of derivations.
    c. Don't consider edits that are identical to a previous edit in the script.
    d. With WIDEN/NARROW, don't consider edits that reverse a previous edit in the script: 
       e.g., WIDEN1RIGHT @ 5, and later NARROW1RIGHT @ 6 (or vice versa).
    
    >>> best_script( \
    ['O',     'B-evt', 'o', 'b-PER', 'I-evt', 'I-evt', 'B-PER', 'O', 'B-ORG', 'I-ORG', 'B-ORG', 'O',     'O',     'O'], \
    ['B-evt', 'I-evt', 'o', 'b-PER', 'I-evt', 'B-PER', 'I-PER', 'O', 'B-LOC', 'I-LOC', 'B-ORG', 'B-evt', 'b-MIS', 'I-evt'])
    (8.0, [(('WIDEN1LEFT', 1, 'evt'), ['B-evt', 'I-evt', 'o', 'b-PER', 'I-evt', 'I-evt', 'B-PER', 'O', 'B-ORG', 'I-ORG', 'B-ORG', 'O', 'O', 'O']), 
    (('NARROW1RIGHT', 5, 'evt'), ['B-evt', 'I-evt', 'o', 'b-PER', 'I-evt', 'O', 'B-PER', 'O', 'B-ORG', 'I-ORG', 'B-ORG', 'O', 'O', 'O']), 
    (('WIDEN1LEFT', 6, 'PER'), ['B-evt', 'I-evt', 'o', 'b-PER', 'I-evt', 'B-PER', 'I-PER', 'O', 'B-ORG', 'I-ORG', 'B-ORG', 'O', 'O', 'O']), 
    (('INSERT', (11, 13), 'evt'), ['B-evt', 'I-evt', 'o', 'b-PER', 'I-evt', 'B-PER', 'I-PER', 'O', 'B-ORG', 'I-ORG', 'B-ORG', 'B-evt', 'o', 'I-evt']), 
    (('INSERT', (12,), 'MIS'), ['B-evt', 'I-evt', 'o', 'b-PER', 'I-evt', 'B-PER', 'I-PER', 'O', 'B-ORG', 'I-ORG', 'B-ORG', 'B-evt', 'b-MIS', 'I-evt']), 
    (('RELABEL', (8, 9), 'ORG', 'LOC'), ['B-evt', 'I-evt', 'o', 'b-PER', 'I-evt', 'B-PER', 'I-PER', 'O', 'B-LOC', 'I-LOC', 'B-ORG', 'B-evt', 'b-MIS', 'I-evt'])])
    >>> best_script( \
    ['O',     'B-evt', 'O',     'B-PER', 'B-evt', 'I-evt', 'B-LOC', 'I-LOC', 'B-PER', 'I-PER'], \
    ['B-evt', 'I-evt', 'B-PER', 'I-PER', 'O',     'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'I-ORG'])
    (9.0, [(('DELETE', (4, 5), 'evt'), ['O', 'B-evt', 'O', 'B-PER', 'O', 'O', 'B-LOC', 'I-LOC', 'B-PER', 'I-PER']), 
    (('WIDEN1LEFT', 3, 'PER'), ['O', 'B-evt', 'B-PER', 'I-PER', 'O', 'O', 'B-LOC', 'I-LOC', 'B-PER', 'I-PER']), 
    (('WIDEN1LEFT', 1, 'evt'), ['B-evt', 'I-evt', 'B-PER', 'I-PER', 'O', 'O', 'B-LOC', 'I-LOC', 'B-PER', 'I-PER']), 
    (('NARROW1RIGHT', 7, 'LOC'), ['B-evt', 'I-evt', 'B-PER', 'I-PER', 'O', 'O', 'B-LOC', 'O', 'B-PER', 'I-PER']), 
    (('WIDEN1LEFT', 6, 'LOC'), ['B-evt', 'I-evt', 'B-PER', 'I-PER', 'O', 'B-LOC', 'I-LOC', 'O', 'B-PER', 'I-PER']), 
    (('WIDEN1LEFT', 8, 'PER'), ['B-evt', 'I-evt', 'B-PER', 'I-PER', 'O', 'B-LOC', 'I-LOC', 'B-PER', 'I-PER', 'I-PER']), 
    (('RELABEL', (5, 6), 'LOC', 'PER'), ['B-evt', 'I-evt', 'B-PER', 'I-PER', 'O', 'B-PER', 'I-PER', 'B-PER', 'I-PER', 'I-PER']),
    (('RELABEL', (7, 8, 9), 'PER', 'ORG'), ['B-evt', 'I-evt', 'B-PER', 'I-PER', 'O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'I-ORG'])])
    >>> best_script( \
    ['O',     'B-evt', 'o',     'b-PER', 'I-evt', 'I-evt', 'B-LOC', 'I-LOC', 'B-PER', 'I-PER'], \
    ['B-evt', 'I-evt', 'B-PER', 'I-PER', 'O',     'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'I-ORG']) # has gap in source only
    (10.0, [(('DELETE', (1, 4, 5), 'evt'), ['O', 'O', 'O', 'B-PER', 'O', 'O', 'B-LOC', 'I-LOC', 'B-PER', 'I-PER']), 
    (('INSERT', (0, 1), 'evt'), ['B-evt', 'I-evt', 'O', 'B-PER', 'O', 'O', 'B-LOC', 'I-LOC', 'B-PER', 'I-PER']), 
    (('WIDEN1LEFT', 3, 'PER'), ['B-evt', 'I-evt', 'B-PER', 'I-PER', 'O', 'O', 'B-LOC', 'I-LOC', 'B-PER', 'I-PER']), 
    (('NARROW1RIGHT', 7, 'LOC'), ['B-evt', 'I-evt', 'B-PER', 'I-PER', 'O', 'O', 'B-LOC', 'O', 'B-PER', 'I-PER']), 
    (('WIDEN1LEFT', 6, 'LOC'), ['B-evt', 'I-evt', 'B-PER', 'I-PER', 'O', 'B-LOC', 'I-LOC', 'O', 'B-PER', 'I-PER']), 
    (('WIDEN1LEFT', 8, 'PER'), ['B-evt', 'I-evt', 'B-PER', 'I-PER', 'O', 'B-LOC', 'I-LOC', 'B-PER', 'I-PER', 'I-PER']), 
    (('RELABEL', (5, 6), 'LOC', 'PER'), ['B-evt', 'I-evt', 'B-PER', 'I-PER', 'O', 'B-PER', 'I-PER', 'B-PER', 'I-PER', 'I-PER']),
    (('RELABEL', (7, 8, 9), 'PER', 'ORG'), ['B-evt', 'I-evt', 'B-PER', 'I-PER', 'O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'I-ORG'])])
    '''
    assert isValidTagging(in_tags)
    assert isValidTagging(out_tags)
    assert len(in_tags)==len(out_tags),'This formulation of edits is over tags, so it requires sequences of equal length'
    
    q = []  # priority queue, accessed by heapq.heappush(heap, cost) heapq.heappop(q)
    heapq.heappush(q, ((0.0, 0, 0, 0.0), [(None, list(in_tags))])) # edit script encoding only the initial state

    best_script = best_script_cost = None

    while q:
        (hcost,anteSplitMerges,dcost,cost),script = heapq.heappop(q) # sequence of (edit, result) pairs. script[-1][1] is the tag sequence after all edits in the script
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
            if edit in prevEdits or (prevEditI-2>firstint(edit[1]) and edit[0]!='RELABEL'):
                #print(edit, script, file=sys.stderr)
                continue # don't repeat a previous edit in this script, and require edits to be mostly left-to-right
            #print(hcost, anteSplitMerges, dcost, cost, zip(*script)[0], edit, file=sys.stderr)
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
            
            heuristic = max(unmatchedTgtChunks - adjacentTgtChunksSameLabel + oMatchedSrcChunks, 
                            unmatchedSrcChunks - adjacentSrcChunksSameLabel + oMatchedTgtChunks)
            # Almost a lower bound on the edit cost: number of chunks on the target side that are not 
            # exactly matched on the source side, plus the number of chunks on the source side that 
            # do not overlap with any chunks on the target side. Also the reverse of source/target. 
            # "Almost" because of SPLIT/MERGE: e.g. B-LOC I-LOC → B-LOC B-LOC, 
            # so there are 2 unmatched chunks on target side but only 1 cost is incurred 
            # with the merge operation. To account for this, we subtract the number of 
            # Begin tags immediately following a chunk with the same label (which could be 
            # MERGEd without additional edits).
            # (We'd also have to correct the other, "O"-matching, term--except e.g. B-LOC B-LOC → O O 
            # is required to be two DELETEs, not a MERGE followed by a DELETE.)
            
            
            nUnmatched = (unmatchedSrcChunks+unmatchedTgtChunks)
            #heapq.heappush(q, ((cost+editcost+heuristic,cost+editcost), script[:]+[(edit,result)]))
            
            # Number of MERGE edits in the script that are followed later by some SPLIT, 
            # possibly in the current edit.
            # (Conflates all MERGEs and all SPLITs, which is slightly hacky: 
            # can probably do better by looking at indices)
            anteSplitMerges = sum(1 for i,e in enumerate(prevEdits) if e[0]=='MERGE' and (edit[0]=='SPLIT' or any(e2[0]=='SPLIT' for e2 in prevEdits[i+1:])))
            
            heapq.heappush(q, ((cost+editcost+heuristic,anteSplitMerges,nUnmatched,cost+editcost), script[:]+[(edit,result)]))
            
    return best_script_cost, best_script[1:]    # don't include the initial state in the returned script

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
        best_script(tuple(pp), tuple(gg))

if __name__=='__main__':
    # running all the doctests takes 10s on my machine
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
