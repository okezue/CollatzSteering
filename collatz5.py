def v2(n):
    if n==0:return 999
    c=0
    while n%2==0:c+=1;n//=2
    return c
def kv5(n):
    c=0;x=(5*n+1)//2
    c+=1
    while x%2!=0:
        x=(5*x+1)//2;c+=1
    return c
def kpv5(n):
    x=n
    for _ in range(kv5(n)):
        x=(5*x+1)//2
    return v2(x)
def apex5(n):
    x=n
    for _ in range(kv5(n)):
        x=(5*x+1)//2
    return x
def kappa5(n):
    x=apex5(n)
    return x>>v2(x)
def check_odd(nmax=10000):
    for n in range(1,nmax,2):
        kn=kappa5(n)
        assert kn%2==1,f"kappa5({n})={kn} is even"
    return True
