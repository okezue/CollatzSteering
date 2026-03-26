from dataclasses import dataclass,field
from typing import List
@dataclass
class C:
    bases:List[int]=field(default_factory=lambda:[32,24,11])
    d:int=512
    nh:int=8
    nel:int=4
    ndl:int=1
    ff:int=2048
    do:float=0.1
    msl:int=16
    bs:int=256
    lr:float=3e-5
    ep:int=1000
    epe:int=300000
    tst:int=100000
    ckf:int=10
    nw:int=4
    seed:int=42
    nmax:int=10**12
    plr:float=1e-3
    pep:int=50
    pts:int=50000
    pvs:int=10000
    kmax:int=16
    sn:int=5000
    tcd:int=4096
    tcl1:float=1e-3
    tcl0:float=1e-4
    tclr:float=1e-4
    tcep:int=50
    dev:str="cuda"
    out:str="output"
