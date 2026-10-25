"""Chunk only the native execution, retaining one phase-1 lease and one selection."""
from .contract import require

def aligned_chunks(start,end,limit,block):
    require(all(type(v) is int for v in (start,end,limit,block)) and
            0<=start<end and limit>=block>0,'invalid phase-1 chunk geometry')
    chunks=[]
    while end-start>limit:
        stop=((start+limit)//block)*block
        require(stop>start,'phase-1 chunk cannot advance')
        chunks.append((start,stop));start=stop
    return chunks+[(start,end)]

def parse_mamba_chunks(raw,rows,attention,start,end,block):
    hybrid=tuple(g for g in range(len(rows)) if g not in attention)
    chunks=[]
    for item in raw:
        require(isinstance(item,(list,tuple)) and len(item)==3,'Mamba chunk shape')
        a,b,more=item
        require(type(a) is int and type(b) is int and a==start and a<b<=end,
                'Mamba chunks must tile [C,e) in order')
        require(isinstance(more,(list,tuple)) and len(more)==len(hybrid),'Mamba chunk group count')
        current=list(rows);n=-(-b//block)
        for g,row in zip(hybrid,more):
            require(isinstance(row,(list,tuple)) and all(type(x) is int for x in row) and
                    len(row)==n and tuple(row[:-1])==(0,)*(n-1) and row[-1]==rows[g][-1],
                    'Mamba chunk must retain its own carry slot with a null head')
            current[g]=tuple(row)
        chunks.append((a,b,tuple(current)));start=b
    require(start==end,'Mamba chunks must end at e')
    return tuple(chunks)


def private_probe_limit(native_limit):
    """Bound private forward activation peaks while the Goal rewriter co-resides.

    The native scheduler capacity is unchanged. Private history/Goal queries
    still consume their entire input, via the existing one-lease chunk plan.
    """
    require(type(native_limit) is int and native_limit>0,'native forward capacity required')
    return min(native_limit,8192)
