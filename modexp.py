P=997
def modexp(a,b,p=P):
    return pow(a,b,p)
def bitlen(b):
    return b.bit_length()
def popcount(b):
    return bin(b).count('1')
def hamming_dist(a,b):
    return bin(a^b).count('1')
def enc10(n):
    if n==0:return[0]
    ds=[]
    while n>0:ds.append(n%10);n//=10
    ds.reverse()
    return ds
def dec10(ds):
    r=0
    for d in ds:r=r*10+d
    return r
