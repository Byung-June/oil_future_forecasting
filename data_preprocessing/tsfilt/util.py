from itertools import islice
import numpy as np

def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield np.array(result)
    for elem in it:
        result = result[1:] + (elem,)
        yield np.array(result)


if __name__=="__main__":
    s = np.random.rand(10)
    print('s', s)
    sw = window(s, 4)
    for x in sw:
        print(x)