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
            return Value(self._v+that._v, self._info+that._info)
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
    def __init__(self, start, end):
        self.start = start
        self.end = end
    def overlap(self, that):
        return set(range(self.start,self.end)) & set(range(that.start,that.end))
    def __repr__(self):
        return 'Span({}, {})'.format(self.start, self.end)
    
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
    (10, ('TNARROWL', (0, 1), 'TSPLIT', 2, 'SNARROWR', (4, 3), 'SDELETE', 2, 'TWIDENL', (11, 9), 'SSPLIT', 11, 'SSPLIT', 12, 'TSPLIT', 13))
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
        
        chart = Chart() # note that assignment to a chart item is a min() operation
        # indexing the chart: source_chunk, target_chunk, right_boundary, source_status, target_status
    
        end_of_block = max(s[b[-1][0]].end, t[b[-1][1]].end)
        for k,(i,j) in enumerate(b):
            # don't align: DELETE or PASS
            if k==0: 
                # - DELETE both s and t
                chart[i,j,None,'DEL','DEL'] = SDELETE(i) + TDELETE(j)
                # - DELETEd target chunk, keep source chunk for a future alignment
                chart[i,j,None,'PASS','DEL'] = TDELETE(j)
                # - DELETEd source chunk, keep target chunk for a future alignment
                chart[i,j,None,'DEL','PASS'] = SDELETE(i)
            else:
                for stat in {'DEL','PASS','ALI'}:
                    # - DELETE source chunk.
                    chart[i,j,None,'DEL',stat] = SDELETE(i) + chart[i-1,j,...,{'DEL','ALI'},stat]
                    # - DELETE target chunk.
                    chart[i,j,None,stat,'DEL'] = TDELETE(j) + chart[i,j-1,...,stat,{'DEL','ALI'}]
                # - DELETEd target chunk, keep source chunk for a future alignment
                chart[i,j,None,'PASS','DEL'] = chart[i-1,j,...,{'DEL','ALI'},'DEL']
                # - DELETEd source chunk, keep target chunk for a future alignment
                chart[i,j,None,'DEL','PASS'] = chart[i,j-1,...,'DEL',{'DEL','ALI'}]
                # - Source XOR target is already aligned; keep the other for a future alignment
                chart[i,j,None,'PASS','ALI'] = chart[i-1,j,...,{'DEL','ALI'},'ALI'] #min(v for key,v in chart[i-1,j].items() if key[-1]=='ALI')
                chart[i,j,None,'ALI','PASS'] = chart[i,j-1,...,'ALI',{'DEL','ALI'}] #min(v for key,v in chart[i,j-1].items() if key[-2]=='ALI')
            # align
            prev_cost = Value() 
            if s[i].start<=t[j].start:
                left_boundary = SNARROWL((s[i].start,t[j].start), t[j].start - s[i].start)
                if k>0:
                    assert b[k-1]==(i,j-1)
                    prev_cost = chart[i,j-1,None,{'ALI','PASS'},{'ALI','DEL'}]
            else:
                left_boundary = TNARROWL((t[j].start,s[i].start), s[i].start - t[j].start)
                if k>0:
                    assert b[k-1]==(i-1,j)
                    prev_cost = chart[i,j-1,None,{'ALI','DEL'},{'ALI','PASS'}]
            min_right = min(s[i].end, t[j].end)
            max_right = max(s[i].end, t[j].end)
            # align and leave open for being the left part of a split, with the original right boundary
            if s[i].end>t[j].end:
                chart[i,j,t[j].end,'ALI','SPL'] = prev_cost + left_boundary
                right_boundary = SNARROWR((max_right,min_right), max_right - min_right)
            else:
                chart[i,j,s[i].end,'SPL','ALI'] = prev_cost + left_boundary
                right_boundary = TNARROWR((max_right,min_right), max_right - min_right)
            chart[i,j,min_right,'ALI','ALI'] = prev_cost + left_boundary + right_boundary
            max_left = max(s[i].start, t[j].start)
            
            #for r in range(min_right,max_right+1):
            if True:
                r = min_right
                # 1-1 align
                #assert s[i].end>r or t[j].end>r, (r, s[i], t[j])
                #right_boundary = SNARROWR((s[i].end,r), s[i].end - r) if s[i].end>t[j].end else TNARROWR((t[j].end,r), t[j].end - r)
                #chart[i,j,r,'ALI','ALI'] = prev_cost + left_boundary + right_boundary
                
                # split
            
                if s[i].start < t[j].start: # SSPLIT: [    i    ] / [J<j] ... [j]
                    for I,J in b[:k]:
                        if I==i:
                            x = chart[i,J,t[J].end,'ALI',{'SPL','ALI'}] \
                                + TWIDENL((t[j].start,t[J].end), t[j].start - t[J].end) \
                                + clearTBetween(J+1, j) \
                                + SSPLIT(t[j].start)
                            if s[i].end>t[j].end:
                                chart[i,j,r,'ALI','SPL'] = x
                            else:
                                chart[i,j,r,'SPL','ALI'] = x
                            chart[i,j,r,'ALI','ALI'] = x + SNARROWR((s[i].end,r), s[i].end - r) + TNARROWR((t[j].end,r), t[j].end - r)
                elif t[j].start < s[i].start: # TSPLIT: [I<i] ... [i] / [    j    ]
                    for I,J in b[:k]:
                        if J==j:
                            x = chart[I,j,s[I].end,{'SPL','ALI'},'ALI'] \
                                + SWIDENL((s[i].start,s[I].end), s[i].start - s[I].end) \
                                + clearSBetween(I+1, i) \
                                + TSPLIT(s[i].start)
                            if s[i].end>t[j].end:
                                chart[i,j,r,'ALI','SPL'] = x
                            else:
                                chart[i,j,r,'SPL','ALI'] = x
                            #pprint(x)
                            #pprint([i,j,r,'ALI','ALI'])
                            chart[i,j,r,'ALI','ALI'] = x + SNARROWR((s[i].end,r), s[i].end - r) + TNARROWR((t[j].end,r), t[j].end - r)
                else:
                    assert k==0
                ''' # old code
                chart0[i,j,s[i].end,'SPL','ALI'] = prev_cost + left_boundary # instead of adjusting right boundary, allow for split point
                chart0[i,j,t[j].end,'ALI','SPL'] = prev_cost + left_boundary # instead of adjusting right boundary, allow for split point
                right_boundary = SNARROWR((s[i].end,r), s[i].end - r) if s[i].end>r else TNARROWR((t[j].end,r), t[j].end - r)
                centry = (i,j,r,'ALI','ALI')
                chart0[centry] = prev_cost + left_boundary + right_boundary
                for split_point in range(max_left, r): # TODO: double-check range
                    mid_boundaryS = SWIDENL((s[i].start,split_point), s[i].start - split_point)
                    mid_boundaryT = TWIDENL((t[j].start,split_point), t[j].start - split_point)
                    for I,J in b[:k]:
                        if J==j:
                            x = TSPLIT(split_point) + mid_boundaryS + right_boundary + chart0[I,J,split_point,{'ALI','SPL'},'ALI'] + clearSBetween(I+1,i)
                            if i==1 and j==0 and r==3:
                                print('~',x)
                                print('~.',chart0[I,J,split_point,{'ALI','SPL'},'ALI'])
                            chart0[centry] = x
                        if I==i:
                            x = SSPLIT(split_point) + mid_boundaryT + right_boundary + chart0[I,J,split_point,'ALI',{'ALI','SPL'}] + clearTBetween(J+1,i)
                            if i==1 and j==0 and r==3:
                                print('~~',x)
                            chart0[centry] = x
                '''
            #print(chart0)
            #chart[i,j,min_right,'ALI','ALI'] = chart0[i,j,min_right,'ALI','ALI']
    
        i,j = b[-1]
        solution += chart[i, j, ..., {'DEL','ALI'}, {'DEL','ALI'}]
    if debug: pprint(chart)
    #print(i,j,chart[i, j, ..., {'DEL','ALI'}, {'DEL','ALI'}])
 
    return solution
 
if __name__=='__main__':
    import doctest
    doctest.testmod()
