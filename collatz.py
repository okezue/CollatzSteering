def v2(n):
    if n==0:return 999
    c=0
    while n%2==0:c+=1;n//=2
    return c
def kv(n):
    return v2(n+1)
def kpv(n):
    k=kv(n);m=(n+1)>>k;a=3**k*m-1
    return v2(a)
def kappa(n):
    k=kv(n);m=(n+1)>>k;a=3**k*m-1
    return a>>v2(a)
def apex(n):
    k=kv(n);m=(n+1)>>k
    return 3**k*m-1
