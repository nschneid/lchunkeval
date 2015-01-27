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
            s = s[:-1] + ', ' + repr(label) + ')'
    
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


def clearSBetween(i,j): return sum((SDELETE(h) for h in range(i,j)), Value())
def clearTBetween(i,j): return sum((TDELETE(h) for h in range(i,j)), Value())


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
        'TSPLIT', 5, 'SRELABEL', (3, '=', '_')))
        
    # 0= 1_ 2~ 3= 4~ 5=  source chunking
    # 0________________  target chunking
    #0  1  2  3  4  5  6  word indices
    >>> s = [Span(0,1,'='),Span(1,2,'_'),Span(2,3,'~'),Span(3,4,'='),Span(4,5,'~'),Span(5,6,'=')]
    >>> t = [Span(0,6,'_')]
    >>> ced(s,t)
    (9, ('SRELABEL', (0, '=', '_'), 'TSPLIT', 1, 'TSPLIT', 2, 
        'SRELABEL', (2, '~', '_'), 'TSPLIT', 3, 'SRELABEL', (3, '=', '~'), 'TSPLIT', 4, 
        'SRELABEL', (4, '~', '='), 'TSPLIT', 5))
    
    # 0= 1_ 2_ 3~ 4_ 5_ 6=  source chunking
    # 0___________________  target chunking
    #0  1  2  3  4  5  6  7 word indices
    >>> s = [Span(0,1,'='),Span(1,2,'_'),Span(2,3,'_'),Span(3,4,'~'),Span(4,5,'_'),Span(5,6,'_'),Span(6,7,'=')]
    >>> t = [Span(0,7,'_')]
    >>> ced(s,t)
    9
    
    # 0=     source chunking
    # 0____  target chunking
    #0  1  2 word indices
    >>> s = [Span(0,1,'=')]
    >>> t = [Span(0,2,'_')]
    >>> ced(s,t)
    (2, ('TNARROWR', (2, 1), 'SRELABEL', (0, '=', '_')))
    
    # 0= 1=  source chunking
    # 0____  target chunking
    #0  1  2 word indices
    >>> s = [Span(0,1,'='),Span(1,2,'=')]
    >>> t = [Span(0,2,'_')]
    >>> ced(s,t)
    (2, ('SRELABEL', (0, '=', '_'), 'TSPLIT', 1))
    
    # 0= 1_ 2= source chunking
    # 0_______  target chunking
    #0  1  2 word indices
    >>> s = [Span(0,1,'='),Span(1,2,'_'),Span(2,3,'_'),Span(3,4,'=')]
    >>> t = [Span(0,4,'_')]
    >>> ced(s,t,debug=True)
    5
    '''
    solution = Value()
    blocks = form_blocks(s, t)
    
    for b in blocks:
        if len(b)==2:
            if b[0] is None:
                solution += TDELETE(b[1])
                continue
            elif b[1] is None:
                solution += SDELETE(b[0])
                continue
        
        bLabels = {lbl for i,j in b for lbl in {s[i].label,t[j].label}}
        
        chart = Chart() # note that assignment to a chart item is a min() operation
        # indexing the chart: source_chunk, target_chunk, right_boundary, source_status, target_status
    
        for k,(i,j) in enumerate(b):
            # don't align: DELETE or PASS
            if k==0: 
                # - DELETE both s and t
                chart[i,j,None,'DEL','DEL',None] = SDELETE(i) + TDELETE(j)
                # - DELETEd target chunk, keep source chunk for a future alignment
                chart[i,j,None,'PASS','DEL',None] = TDELETE(j)
                # - DELETEd source chunk, keep target chunk for a future alignment
                chart[i,j,None,'DEL','PASS',None] = SDELETE(i)
            else:
                for stat in {'DEL','PASS','ALI'}:
                    # - DELETE source chunk.
                    chart[i,j,None,'DEL',stat,(t[j].label if stat=='ALI' else None)] = SDELETE(i) + chart[i-1,j,...,{'DEL','ALI'},stat,...]
                    # - DELETE target chunk.
                    chart[i,j,None,stat,'DEL',(s[i].label if stat=='ALI' else None)] = TDELETE(j) + chart[i,j-1,...,stat,{'DEL','ALI'},...]
                # - DELETEd target chunk, keep source chunk for a future alignment
                chart[i,j,None,'PASS','DEL',None] = chart[i-1,j,...,{'DEL','ALI'},'DEL',...]
                # - DELETEd source chunk, keep target chunk for a future alignment
                chart[i,j,None,'DEL','PASS',None] = chart[i,j-1,...,'DEL',{'DEL','ALI'},...]
                # - Source XOR target is already aligned; keep the other for a future alignment
                # TODO: can we just use observed labels, or do we need to store entries for all labels?
                chart[i,j,None,'PASS','ALI',None] = chart[i-1,j,...,{'DEL','ALI'},'ALI',...]
                chart[i,j,None,'ALI','PASS',None] = chart[i,j-1,...,'ALI',{'DEL','ALI'},...]
            # align
            prev_cost = lambda lbl: Value()
            if s[i].start<=t[j].start:
                left_boundary = SNARROWL((s[i].start,t[j].start), t[j].start - s[i].start)
                if k>0:
                    assert b[k-1]==(i,j-1)
                    prev_cost = lambda lbl: chart[i,j-1,None,{'ALI','PASS'},{'ALI','DEL'},None] #+ SRELABEL((i,lbl,s[i].label),lbl,s[i].label)
            else:
                left_boundary = TNARROWL((t[j].start,s[i].start), s[i].start - t[j].start)
                if k>0:
                    assert b[k-1]==(i-1,j)
                    prev_cost = lambda lbl: chart[i-1,j,None,{'ALI','DEL'},{'ALI','PASS'},None] #+ TRELABEL((j,lbl,t[j].label),lbl,t[j].label)
            
            min_right = min(s[i].end, t[j].end)
            max_right = max(s[i].end, t[j].end)
            max_left = max(s[i].start, t[j].start)
            
            # align and leave open for being the left part of a split, with the original right boundary
            for lbl in bLabels:
                relabel_cost = SRELABEL((i,s[i].label,lbl),s[i].label,lbl) + TRELABEL((j,t[j].label,lbl),t[j].label,lbl)
                if s[i].end>t[j].end:
                    chart[i,j,t[j].end,'ALI','SPL',lbl] = prev_cost(lbl) + left_boundary + relabel_cost # relabel target, then split
            #        chart[i,j,t[j].end,'ALI','SPL',t[j].label] = prev_cost(lbl) + left_boundary + relabel_cost # split target, then relabel left portion
            #        if j==0: assert False,(i,j,t[j].end,t[j].label)
                    right_boundary = SNARROWR((max_right,min_right), max_right - min_right)
                else:
                    chart[i,j,s[i].end,'SPL','ALI',lbl] = prev_cost(lbl) + left_boundary + relabel_cost # relabel, then split
                    if s[i].end==5 and lbl=='_':
                        assert True or False,(prev_cost(lbl), left_boundary, relabel_cost)
            #        chart[i,j,s[i].end,'SPL','ALI',s[i].label] = prev_cost(lbl) + left_boundary + relabel_cost # split, then relabel
                    right_boundary = TNARROWR((max_right,min_right), max_right - min_right)
                chart[i,j,min_right,'ALI','ALI',lbl] = prev_cost(lbl) + left_boundary + right_boundary \
                    + relabel_cost
            
            
                #for r in range(min_right,max_right+1):
                for plbl in bLabels:
                    r = min_right
                    # 1-1 align
                    #assert s[i].end>r or t[j].end>r, (r, s[i], t[j])
                    #right_boundary = SNARROWR((s[i].end,r), s[i].end - r) if s[i].end>t[j].end else TNARROWR((t[j].end,r), t[j].end - r)
                    #chart[i,j,r,'ALI','ALI'] = prev_cost + left_boundary + right_boundary
                
                    # split
                    for I,J in b[:k]:
                        x = None
                        if I==i and s[i].start < t[j].start: # SSPLIT: [    i    ] / [J<j] ... [j]
                                x = chart[i,J,t[J].end,'ALI',{'SPL','ALI'},plbl] \
                                    + TRELABEL((j,plbl,lbl), plbl, lbl) \
                                    + TWIDENL((t[j].start,t[J].end), t[j].start - t[J].end) \
                                    + clearTBetween(J+1, j) \
                                    + SSPLIT(t[j].start)
                        elif J==j and t[j].start < s[i].start: # TSPLIT: [I<i] ... [i] / [    j    ]
                                x = chart[I,j,s[I].end,{'SPL','ALI'},'ALI',lbl] \
                                    + SRELABEL((i,plbl,lbl), plbl, lbl) \
                                    + SWIDENL((s[i].start,s[I].end), s[i].start - s[I].end) \
                                    + clearSBetween(I+1, i) \
                                    + TSPLIT(s[i].start)
                        
                        if x:
                            if s[i].end>t[j].end:
                                chart[i,j,r,'ALI','SPL',lbl] = x + TRELABEL((j,t[j].label,lbl), t[j].label, lbl)    # relabel target, then split
                                chart[i,j,r,'ALI','SPL',t[j].label] = x + TRELABEL((j,t[j].label,lbl), t[j].label, lbl) # split target, then relabel left side
                            else:
                                chart[i,j,r,'SPL','ALI',lbl] = x + SRELABEL((i,s[i].label,lbl), s[i].label, lbl)    # relabel source, then split
                                chart[i,j,r,'SPL','ALI',s[i].label] = x + SRELABEL((i,s[i].label,lbl), s[i].label, lbl) # split source, then relabel left side
                            chart[i,j,r,'ALI','ALI',lbl] = x + SNARROWR((s[i].end,r), s[i].end - r) + TNARROWR((t[j].end,r), t[j].end - r)
                
            #print(chart0)
            #chart[i,j,min_right,'ALI','ALI'] = chart0[i,j,min_right,'ALI','ALI']
    
        i,j = b[-1]
        solution += chart[i, j, ..., {'DEL','ALI'}, {'DEL','ALI'}, ...]
        if debug: pprint(chart)
    #print(i,j,chart[i, j, ..., {'DEL','ALI'}, {'DEL','ALI'}])
 
    return solution
 
if __name__=='__main__':
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
