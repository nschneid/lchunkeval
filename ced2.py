#!/usr/bin/env python2.7
#coding=utf-8
'''
Chunking Edit Distance (CED): represents the relationship between two chunkings 
of a sentence. A chunking assigns each word of the sentence to 0 or 1 contiguous 
units (chunks). Each chunk bears a categorical label. If the source and target 
chunkings are not identical, their relationship is characterized by edit operations 
such as inserting, deleting, and relabeling entire chunks, 
narrowing/widening a chunk's span on the left or right, 
splitting a chunk into two chunks, and merging two adjacent chunks with identical labels. 
If a particular chunk on the source side is matched with/transformed into a chunk on the 
target side, they are said to be aligned.
Apart from insertions and deletions, alignments are only possible between overlapping 
source/target chunks.

TODO: labels not yet supported

@author: Nathan Schneider (nschneid@inf.ed.ac.uk)
@since: 2015-01-22
'''
from __future__ import print_function
from pprint import pprint

def partial_match(query):
    '''
    >>> partial_match((0,1,2))((0,1,2))
    True
    >>> partial_match((0,1,3))((0,1,2))
    False
    >>> partial_match((0, 0, Ellipsis, {'DEL','ALI'}, {'DEL','ALI'}))((0, 0, 2, 'ALI', 'ALI'))
    True
    >>> partial_match((0, 0, Ellipsis, {'DEL','ALI'}, {'DEL','ALI'}))((0, 0, 2, 'ALI', None))
    False
    '''
    def _(key):
        assert len(key)==len(query)
        for k,q in zip(key,query):
            if q is not Ellipsis:
                if isinstance(q,set):
                    if k not in q:
                        return False
                elif k!=q:
                    return False
        return True
    return _

NO_DEFAULT = object()

class Value(object):
    def __init__(self, v=0, info=()):
        self._v = v
        self._info = info
    def __add__(self, that):
        if that._v!=0:
            if self._v!=0:
                return Value(self._v+that._v, self._info+that._info)
            return Value(that._v, that._info)
        return Value(self._v, self._info)
    def __repr__(self):
        return repr((self._v,self._info))

class Chart(dict):
    def __setitem__(self, k, v, op=min):
        if k in self:
            oldv = dict.__getitem__(self, k)
            dict.__setitem__(self, k, op(v, oldv, key=lambda x: x._v))
        else:
            dict.__setitem__(self, k, v)

    def lookup(self, k, op=min, default=NO_DEFAULT):
        '''Return (key,value) pair for best (according to 'op') entry currently in the chart 
        whose key matches tuple 'k' (Ellipsis fields in 'k' match anything). 
        'op' must be min, max, or similar (take an iterable arg and a 'key' parameter).'''
    
        if Ellipsis not in k and not any(isinstance(x,set) for x in k):
            return (k,dict.__getitem__(self,k))
        matches = [(ky,dict.__getitem__(self,ky)) for ky in filter(partial_match(k), self.keys())]
        if default is NO_DEFAULT:
            return op(matches, key=lambda x: x[1]._v)
        return op(matches, key=lambda x: x[1]._v) if matches else default
    
    def __getitem__(self, k, semi0=Value(float('inf')), op=min):
        try:
            kv = self.lookup(k,op=op)
        except ValueError:
            return semi0
        return kv[1]



class Span(object):
    def __init__(self, start, end, label=''):
        self.start = start
        self.end = end
        self.label = label
    def overlap(self, that):
        return set(range(self.start,self.end)) & set(range(that.start,that.end))
    def __repr__(self):
        s = 'Span({}, {})'.format(self.start, self.end)
        if self.label:
            s = s[:-1] + ', ' + repr(self.label) + ')'
        return s
    
# edit costs

def editop(f):
    def _(at, *costargs):
        cost = f(*costargs)
        name = f.func_name
        assert cost>=0,(name,cost,at)
        return Value(cost,(name,at))
    return _

@editop
def SDELETE(): return 2 # a.k.a. DELETE
@editop
def TDELETE(): return 2 # a.k.a. INSERT
@editop
def SSPLIT(): return 1 # a.k.a. SPLIT
@editop
def TSPLIT(): return 1 # a.k.a. MERGE
@editop
def SNARROWL(width): return width # a.k.a. NARROWL
@editop
def SNARROWR(width): return width # a.k.a. NARROWR
@editop
def TNARROWL(width): return width # a.k.a. WIDENL
@editop
def TNARROWR(width): return width # a.k.a. WIDENR
@editop
def SWIDENL(width): return width # a.k.a. WIDENL - only for split points
@editop
def TWIDENL(width): return width # a.k.a. NARROWL - only for split points
@editop
def SRELABEL(fromL,toL): return int(fromL!=toL)
@editop
def TRELABEL(fromL,toL): return int(fromL!=toL)




class Alignment(object):
    '''
    Represents a candidate alignment between an overlapping (soure chunk, target chunk) pair.
    Each block is a doubly linked list of Alignment instances.
    '''
    
    chart = None
    
    def __init__(self, sChunkI, sChunk, tChunkI, tChunk, prevAlign=None):
        self.sChunk = sChunk
        self.i = sChunkI    # offset in list of source chunks
        self.tChunk = tChunk
        self.j = tChunkI    # offset in list of target chunks
        self.prevAlign = prevAlign
        self.nextAlign = None
        if self.prevAlign:
            self.prevAlign.nextAlign = self
        self._side = None
    
    def __repr__(self):
        return '<' + repr(self.sChunk) + ' / ' + repr(self.tChunk) + '>'
    
    def __getitem__(self, (r, status1, status2, lbl)):
        if self.side=='s':
            return self.chart[self.i,self.j, r, status1,status2, lbl]
        else:
            return self.chart[self.i,self.j, r, status2,status1, lbl]
    
    def __setitem__(self, (r, status1, status2, lbl), v):
        if self.side=='s':
            self.chart[self.i,self.j, r, status1,status2, lbl] = v
        else:
            self.chart[self.i,self.j, r, status2,status1, lbl] = v
    
    @property
    def side(self):
        return self._side
    @side.setter
    def side(self,v):
        assert v in {'s','t'}
        self._side = v
        self.oside = 't' if v=='s' else 's'
    
    def chunk(self, side):
        if side=='s':
            return self.sChunk
        elif side=='t':
            return self.tChunk
        raise ValueError("'side' must be 's' (source) or 't' (target)")
    
    def prevAlignForChunk(self, side):
        assert side in {'s','t'}
        if self.prevAlign:
            if self.prevAlign.chunk(side) is self.chunk(side):
                self.prevAlign.side = side # ensure chart indexing is in the correct direction
                return self.prevAlign
    
    def allPrevAlignsForChunk(self, side):
        p = self.prevAlignForChunk(side)
        if not p:
            return []
        return p.allPrevAlignsForChunk(side)+[p]
    
    def nextAlignForChunk(self, side):
        assert side in {'s','t'}
        if self.nextAlign:
            if self.nextAlign.chunk(side) is self.chunk(side):
                self.nextAlign.side = side # ensure chart indexing is in the correct direction
                return self.nextAlign
        return None
    
    def allNextAlignsForChunk(self, side):
        n = self.nextAlignForChunk(side)
        if not n:
            return []
        return n.allNextAlignsForChunk(side)+[n]
    
    def possibleLabelsForChunk(self, side):
        '''Labels seen for the chunk or one of its potentially aligned chunks
        
        >>> s = [Span(0,1,'='), Span(1,2,'~'), Span(2,3,'~')]
        >>> t = [Span(0,3,'_')]
        >>> aa = [Alignment(0, s[0], 0, t[0])]
        >>> aa.append(Alignment(1, s[1], 0, t[0], aa[-1]))
        >>> aa.append(Alignment(2, s[2], 0, t[0], aa[-1]))
        >>> aa
        [<Span(0, 1, '=') / Span(0, 3, '_')>, <Span(1, 2, '~') / Span(0, 3, '_')>, <Span(2, 3, '~') / Span(0, 3, '_')>]
        >>> sorted(aa[0].possibleLabelsForChunk('s'))
        ['=', '_']
        >>> sorted(aa[0].possibleLabelsForChunk('t'))
        ['=', '_', '~']
        >>> aa[0].possibleLabelsForChunk('t')==aa[1].possibleLabelsForChunk('t')==aa[2].possibleLabelsForChunk('t')
        True
        >>> sorted(aa[1].possibleLabelsForChunk('s'))
        ['_', '~']
        >>> aa[1].possibleLabelsForChunk('s')==aa[2].possibleLabelsForChunk('s')
        True
        '''
        oside = 't' if side=='s' else 's'
        return {self.chunk(side).label, self.chunk(oside).label} \
            | {a.chunk(oside).label for a in self.allPrevAlignsForChunk(side)} \
            | {a.chunk(oside).label for a in self.allNextAlignsForChunk(side)}
    
    def DELETE(self, side=None):
        if side is None: side = self.side
        return SDELETE(self.i) if side=='s' else TDELETE(self.j)
    
    def WIDENL(self, side, to):
        if side=='s':
            return SWIDENL((self.sChunk.start,to), self.sChunk.start - to)
        return     TWIDENL((self.tChunk.start,to), self.tChunk.start - to)
    
    def NARROWL(self):
        sStart, tStart = self.sChunk.start, self.tChunk.start
        if sStart<=tStart:
            return SNARROWL((sStart,tStart), tStart - sStart)
        return     TNARROWL((tStart,sStart), sStart - tStart)
    
    def NARROWR(self):
        sEnd, tEnd = self.sChunk.end, self.tChunk.end
        if sEnd>=tEnd:
            return SNARROWR((sEnd,tEnd), sEnd - tEnd)
        return     TNARROWR((tEnd,sEnd), tEnd - sEnd)
    
    def SPLIT(self):
        if self.side=='s':
            return SSPLIT(self.tChunk.start)
        return     TSPLIT(self.sChunk.start)
    
    def RELABEL_BOTH(self, lbl):
        return SRELABEL((self.i,self.sChunk.label,lbl),self.sChunk.label,lbl) \
             + TRELABEL((self.j,self.tChunk.label,lbl),self.tChunk.label,lbl)
             
    def RELABEL(self, side, plbl, lbl):
        if side=='s':
            return SRELABEL((self.i,plbl,lbl), plbl, lbl)
        return     TRELABEL((self.j,plbl,lbl), plbl, lbl)
    
    def CLEAR_FROM(self, earlierAlign):
        assert (self.sChunk is earlierAlign.sChunk) ^ (self.tChunk is earlierAlign.tChunk)
        side = 's' if self.sChunk is earlierAlign.sChunk else 't'
        allprev = self.allPrevAlignsForChunk(side)
        return sum((a2.DELETE(side) for a2 in allprev[allprev.index(earlierAlign)+1:]), Value())

def form_blocks(s, t):
    '''
    A block is a minimal span of the sentence containing ≥1 source and/or target chunks
    such that there is a clean break between chunks on the left and right for 
    both the source and target sides. This function is a generator over candidate 
    source-to-target chunk alignments, grouped by block. Candidate alignments 
    are all (source,target) pairs whose chunks overlap by at least 1 token. 
    Blocks with no overlap (only 1 chunk) are represented as (source,None) or (None,target).
    
    # - 0_ 1_____ 2_ 3_ -  -  4_____________ 5_   source chunking
    # 0_______ -  -  1_ -  -  2_ -  -  3_ 4____   target chunking
    #0 1  2   3  4  5  6  7  8  9 10 11  12 13 14 word indices
    #[===========|==|==]     [=================]  the 4 blocks implied by source & target chunking
    >>> s = [Span(1,2),Span(2,4),Span(4,5),Span(5,6),Span(8,13),Span(13,14)]
    >>> t = [Span(0,3),Span(5,6),Span(8,9),Span(11,12),Span(12,14)]
    >>> list(form_blocks(s,t))
    [[(0, 0), (1, 0)], (2, None), [(3, 1)], [(4, 2), (4, 3), (4, 4), (5, 4)]]
    '''
    i = 0
    j = 0
    
    while True:
        if i>=len(s):
            while j<len(t):
                yield (None,j)
                j += 1
            break
        elif j>=len(t):
            while i<len(s):
                yield (i,None)
                i += 1
            break
        elif not s[i].overlap(t[j]):
            if s[i].start<t[j].start:
                yield (i,None)
                i += 1
            else:
                yield (None,j)
                j += 1
        else:
            b = [(i,j)]
            while True:
                if j+1<len(t) and s[b[-1][0]].overlap(t[j+1]):
                    j += 1
                    b.append((i,j))
                elif i+1<len(s) and t[b[-1][1]].overlap(s[i+1]):
                    i += 1
                    b.append((i,j))
                else:
                    break
            yield b
            i += 1
            j += 1

def ced(s, t, debug=False):
    '''
    Returns a lowest-cost derivation for the Chunking Edit Distance 
    between a source chunking 's' and target 't'. 
    Dynamic programming.
    
    # - 0_ 1_____ 2_ 3_ -  -  4_____________ 5_   source chunking
    # 0_______ -  -  1_ -  -  2_ -  -  3_ 4____   target chunking
    #0 1  2   3  4  5  6  7  8  9 10 11  12 13 14 word indices
    >>> s = [Span(1,2),Span(2,4),Span(4,5),Span(5,6),Span(8,13),Span(13,14)]
    >>> t = [Span(0,3),Span(5,6),Span(8,9),Span(11,12),Span(12,14)]
    >>> ced(s,t)
    (10, ('TNARROWL', (0, 1), 'TSPLIT', 2, 'SNARROWR', (4, 3), 'SDELETE', 2, 
        'TWIDENL', (11, 9), 'SSPLIT', 11, 'SSPLIT', 12, 'TSPLIT', 13))
    
    # - 0_ 1_____ 2_ 3= -  -  4_____________ 5_   source chunking
    # 0======= -  -  1_ -  -  2= -  -  3_ 4____   target chunking
    #0 1  2   3  4  5  6  7  8  9 10 11  12 13 14 word indices
    >>> s = [Span(1,2,'_'),Span(2,4,'_'),Span(4,5,'_'),Span(5,6,'='),Span(8,13,'_'),Span(13,14,'_')]
    >>> t = [Span(0,3,'='),Span(5,6,'_'),Span(8,9,'='),Span(11,12,'_'),Span(12,14,'_')]
    >>> ced(s,t)
    (13, ('TNARROWL', (0, 1), 'TRELABEL', (0, '=', '_'), 'TSPLIT', 2, 'SNARROWR', (4, 3), 
        'SDELETE', 2, 'SRELABEL', (3, '=', '_'), 'TRELABEL', (2, '=', '_'), 
        'TWIDENL', (11, 9), 'SSPLIT', 11, 'SSPLIT', 12, 'TSPLIT', 13))
    
    # 0_ 1===== 2____ 3=   source chunking
    # 0=========== 1____   target chunking
    #0  1  2   3  4  5  6  word indices
    >>> s = [Span(0,1,'_'),Span(1,3,'='),Span(3,5,'_'),Span(5,6,'=')]
    >>> t = [Span(0,4,'='),Span(4,6,'_')]
    >>> ced(s,t)
    (6, ('SRELABEL', (0, '_', '='), 'TSPLIT', 1, 'TNARROWR', (4, 3), 'SNARROWL', (3, 4), 
        'SRELABEL', (3, '=', '_'), 'TSPLIT', 5))
        
    # 0= 1_ 2~ 3= 4~ 5=  source chunking
    # 0________________  target chunking
    #0  1  2  3  4  5  6  word indices
    >>> s = [Span(0,1,'='),Span(1,2,'_'),Span(2,3,'~'),Span(3,4,'='),Span(4,5,'~'),Span(5,6,'=')]
    >>> t = [Span(0,6,'_')]
    >>> ced(s,t)
    (10, ('SRELABEL', (0, '=', '_'), 'TSPLIT', 1, 'SRELABEL', (2, '~', '_'), 'TSPLIT', 2, 'SRELABEL', (3, '=', '_'), 'TSPLIT', 3, 'SRELABEL', (4, '~', '_'), 'TSPLIT', 4, 'SRELABEL', (5, '=', '_'), 'TSPLIT', 5))
    
    # 0= 1_ 2_ 3~ 4_ 5_ 6=  source chunking
    # 0___________________  target chunking
    #0  1  2  3  4  5  6  7 word indices
    >>> s = [Span(0,1,'='),Span(1,2,'_'),Span(2,3,'_'),Span(3,4,'~'),Span(4,5,'_'),Span(5,6,'_'),Span(6,7,'=')]
    >>> t = [Span(0,7,'_')]
    >>> ced(s,t)
    (9, ('SRELABEL', (0, '=', '_'), 'TSPLIT', 1, 'TSPLIT', 2, 'SRELABEL', (3, '~', '_'), 'TSPLIT', 3, 'TSPLIT', 4, 'TSPLIT', 5, 'SRELABEL', (6, '=', '_'), 'TSPLIT', 6))
    
    # 0=     source chunking
    # 0____  target chunking
    #0  1  2 word indices
    >>> s = [Span(0,1,'=')]
    >>> t = [Span(0,2,'_')]
    >>> ced(s,t)
    (2, ('TNARROWR', (2, 1), 'SRELABEL', (0, '=', '_')))
    
    # 0_ 1_  source chunking
    # 0____  target chunking
    #0  1  2 word indices
    >>> s = [Span(0,1,'_'),Span(1,2,'_')]
    >>> t = [Span(0,2,'_')]
    >>> ced(s,t)
    (1, ('TSPLIT', 1))
    
    # 0= 1=  source chunking
    # 0____  target chunking
    #0  1  2 word indices
    >>> s = [Span(0,1,'='),Span(1,2,'=')]
    >>> t = [Span(0,2,'_')]
    >>> ced(s,t)
    (2, ('TRELABEL', (0, '_', '='), 'TSPLIT', 1))
    
    # 0_ 1=  source chunking
    # 0____  target chunking
    #0  1  2 word indices
    >>> s = [Span(0,1,'_'),Span(1,2,'=')]
    >>> t = [Span(0,2,'_')]
    >>> ced(s,t)
    (2, ('SRELABEL', (1, '=', '_'), 'TSPLIT', 1))
    
    # 0= 1_  source chunking
    # 0____  target chunking
    #0  1  2 word indices
    >>> s = [Span(0,1,'='),Span(1,2,'_')]
    >>> t = [Span(0,2,'_')]
    >>> ced(s,t)
    (2, ('SRELABEL', (0, '=', '_'), 'TSPLIT', 1))
    
    # 0= 1_ 2=  source chunking
    # 0_______  target chunking
    #0  1  2  3 word indices
    >>> s = [Span(0,1,'='),Span(1,2,'_'),Span(2,3,'=')]
    >>> t = [Span(0,3,'_')]
    >>> ced(s,t)
    (4, ('SRELABEL', (0, '=', '_'), 'TSPLIT', 1, 'SRELABEL', (2, '=', '_'), 'TSPLIT', 2))
    
    # 0= 1~ 2~  source chunking
    # 0_______  target chunking
    #0  1  2  3 word indices
    >>> s = [Span(0,1,'='),Span(1,2,'~'),Span(2,3,'~')]
    >>> t = [Span(0,3,'_')]
    >>> ced(s,t)
    (5, ('SRELABEL', (0, '=', '_'), 'SRELABEL', (1, '~', '_'), 'TSPLIT', 1, 'SRELABEL', (2, '~', '_'), 'TSPLIT', 2))
    
    #    0= 1~  source chunking
    # 0_______  target chunking
    #0  1  2  3 word indices
    >>> s = [Span(1,2,'='),Span(2,3,'~')]
    >>> t = [Span(0,3,'_')]
    >>> ced(s,t)
    (4, ('TNARROWL', (0, 1), 'SRELABEL', (0, '=', '_'), 'SRELABEL', (1, '~', '_'), 'TSPLIT', 2))
    
    #    0= 1=  source chunking
    # 0_______  target chunking
    #0  1  2  3 word indices
    >>> s = [Span(1,2,'='),Span(2,3,'=')]
    >>> t = [Span(0,3,'_')]
    >>> ced(s,t)
    (3, ('TNARROWL', (0, 1), 'TRELABEL', (0, '_', '='), 'TSPLIT', 2))
    
    #    0= 1_  source chunking
    # 0_______  target chunking
    #0  1  2  3 word indices
    >>> s = [Span(1,2,'='),Span(2,3,'_')]
    >>> t = [Span(0,3,'_')]
    >>> ced(s,t)
    (3, ('TNARROWL', (0, 1), 'SRELABEL', (0, '=', '_'), 'TSPLIT', 2))
    '''
    solution = Value()
    derivation = []
    blocks = form_blocks(s, t)
    
    for b in blocks:
        if len(b)==2:
            if b[0] is None:
                solution += TDELETE(b[1])
                continue
            elif b[1] is None:
                solution += SDELETE(b[0])
                continue
        
        aa = [Alignment(b[0][0], s[b[0][0]], b[0][1], t[b[0][1]])]
        for i,j in b[1:]:
            aa.append(Alignment(i, s[i], j, t[j], aa[-1]))
        
        bLabels = {lbl for i,j in b for lbl in {s[i].label,t[j].label}}
        
        # note that assignment to a chart item is a min() operation
        # indexing the chart: source_chunk, target_chunk, right_boundary, source_status, target_status, ending_label
        
        Alignment.chart = Chart()   # new chart for each block
        
        if debug: print(aa)
        
        for k,a in enumerate(aa):
        
            max_left = max(a.sChunk.start, a.tChunk.start)
            min_right = min(a.sChunk.end, a.tChunk.end)
            
            left_boundary = a.NARROWL()
            right_boundary = a.NARROWR()
        
            
            for side,oside in (('s','t'),('t','s')):
                a.side = side
                
                # don't align: DELETE or PASS
                
                if a.prevAlignForChunk(oside):
                    # - DELETE source chunk.
                    a[None,'DEL','DEL',None] = a.DELETE() + a.prevAlignForChunk(oside)[...,{'DEL','ALI'},'DEL',...]
                    a[None,'DEL','PASS',None] = a.DELETE() + a.prevAlignForChunk(oside)[...,{'DEL','ALI'},'PASS',...]
                    a[None,'DEL','ALI',a.chunk(oside).label] = a.DELETE() + a.prevAlignForChunk(oside)[...,{'DEL','ALI'},'ALI',...]
                    # - DELETEd target chunk, keep source chunk for a future alignment
                    a[None,'PASS','DEL',None] = a.prevAlignForChunk(oside)[...,{'DEL','ALI'},'DEL',...]
                    # - Source XOR target is already aligned; keep the other for a future alignment
                    # TODO: can we just use observed labels, or do we need to store entries for all labels?
                    a[None,'PASS','ALI',None] = a.prevAlignForChunk(oside)[...,{'DEL','ALI'},'ALI',...]
                elif k==0:    # if k were >0, we'd need to account for what happened up to here
                    # - DELETE both s and t
                    a[None,'DEL','DEL',None] = a.DELETE('s') + a.DELETE('t')
                    # - DELETEd source chunk, keep target chunk for a future alignment
                    a[None,'DEL','PASS',None] = a.DELETE()
                    
                # align
            
                if a.prevAlignForChunk(oside):
                    assert k>0
                    prev_cost = a.prevAlignForChunk(oside)[None,{'ALI','PASS'},{'ALI','DEL'},None] #+ SRELABEL((i,lbl,s[i].label),lbl,s[i].label)
                elif k==0:
                    prev_cost = Value()
                else:
                    prev_cost = Value(float('inf'),())
                
                # align and leave open for being the left part of a split, with the original right boundary
                for lbl in a.possibleLabelsForChunk(oside):
                    relabel_cost = a.RELABEL_BOTH(lbl)
                    if a.chunk(oside).end <= a.chunk(side).end:
                        a[a.chunk(oside).end,'ALI','SPL',lbl] = prev_cost + left_boundary + relabel_cost
                    a[min_right,'ALI','ALI',lbl] = prev_cost + left_boundary + right_boundary + relabel_cost
            
                    #for plbl in bLabels:
                    if True:
                        # split
                        for p in a.allPrevAlignsForChunk(side):
                            # SPLIT side / oside: [    A @i    ] / [B @J<j] ... [C @j]
                            x = p[p.chunk(oside).end,'ALI',{'SPL','ALI'},lbl]    # ...[ A ] / ...[ B ]
                                   #+ p.RELABEL(oside, plbl, lbl) \
                            x      += a.RELABEL(oside, a.chunk(oside).label, lbl) # [ C ]
                            x      += a.WIDENL(oside, p.chunk(oside).end) \
                                    + a.CLEAR_FROM(p) \
                                    + a.SPLIT()    # B]...[C
                            
                            chk = a.chunk(side) # A
                            if chk.end < a.chunk(oside).end:    # A] . / . C]
                                a[min_right,'SPL','ALI',lbl] = x
                                #a[min_right,'SPL','ALI',lbl] = x + a.RELABEL(side, chk.label, lbl)    # relabel source, then split
                                #a[min_right,'SPL','ALI',chk.label] = x + a.RELABEL(side, chk.label, lbl) # split source, then relabel left side
                            elif a.chunk(oside).end < chk.end:
                                a[min_right,'ALI','SPL',lbl] = x
                            a[min_right,'ALI','ALI',lbl] = x + a.NARROWR()
        
        # a is leftover from the end of the loop
        derivation.append(Alignment.chart.lookup((a.i, a.j, Ellipsis, {'DEL','ALI'}, {'DEL','ALI'}, Ellipsis))[0])
        #derivation.append(prev_cost)
        solution += a[..., {'DEL','ALI'}, {'DEL','ALI'}, ...]
        if debug:
            pprint(Alignment.chart)
            print(derivation)
 
    return solution
 
if __name__=='__main__':
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
