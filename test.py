import torch,os,sys
sys.path.insert(0,os.path.dirname(__file__))
from config import C
from collatz import v2,kv,kpv,kappa,apex
from data import enc,dec,CStream,CFixed,collate,BOS,EOS,PAD,VSZ
from model import CTF
def test_collatz():
    assert v2(1)==0;assert v2(2)==1;assert v2(8)==3;assert v2(12)==2
    assert kv(1)==1;assert kv(3)==2;assert kv(7)==3;assert kv(15)==4
    assert kappa(1)==1
    assert kappa(3)==1
    assert kappa(5)==1
    assert kappa(7)==13
    assert kappa(9)==7
    assert kappa(11)==13
    assert kappa(13)==5
    assert kappa(15)==5
    assert kpv(1)==1;assert kpv(7)==1;assert kpv(13)==2
    for n in range(1,200,2):
        k=kv(n);kp=kpv(n);kn=kappa(n)
        m=(n+1)>>k;a=3**k*m-1
        assert a>>kp==kn
        assert kn%2==1
    print("collatz: PASS")
def test_enc_dec():
    for b in[2,10,11,24,32]:
        for n in[0,1,7,100,999999]:
            d=enc(n,b,40)
            assert dec(d,b)==n,f"fail b={b} n={n}"
    for b in[10,24,32]:
        d=enc(123456789,b,40)
        assert dec(d,b)==123456789,f"fail b={b} n=123456789"
    assert enc(255,2,20)==[1,1,1,1,1,1,1,1]
    assert enc(31,32,20)==[31]
    assert enc(32,32,20)==[1,0]
    print("enc/dec: PASS")
def test_model():
    cfg=C(d=64,nh=4,nel=2,ndl=1,ff=128,msl=12,dev='cpu')
    m=CTF(cfg,32)
    s=torch.randint(0,35,(4,8));t=torch.randint(0,35,(4,6))
    sm=torch.ones(4,8,dtype=torch.bool);tm=torch.ones(4,6,dtype=torch.bool)
    out=m(s,t,sm,tm)
    assert out.shape==(4,6,35),f"got {out.shape}"
    out2,hs=m(s,t,sm,tm,rh=True)
    assert out2.shape==out.shape
    assert len(hs)==3
    assert hs[0].shape==(4,8,64)
    gen=m.generate(s,sm,ml=10)
    assert gen.shape[0]==4
    assert gen.shape[1]<=10
    steer={1:torch.randn(1,1,64)}
    gen2=m.generate(s,sm,ml=10,steer=steer)
    assert gen2.shape[0]==4
    hs2=m.enc_acts(s,sm)
    assert len(hs2)==3
    print("model: PASS")
def test_data():
    from functools import partial
    ds=CFixed(100,32,12,10**6,seed=42)
    assert len(ds)==100
    item=ds[0]
    assert'n'in item and'kn'in item and'k'in item
    cf=partial(collate,base=32,msl=12)
    dl=torch.utils.data.DataLoader(ds,batch_size=16,collate_fn=cf)
    bat=next(iter(dl))
    assert bat['s'].shape[0]==16
    assert bat['sm'].shape==bat['s'].shape
    assert bat['k'].shape==(16,)
    stream=CStream(32,12,10**6,seed=42)
    it=iter(stream)
    for _ in range(5):
        item=next(it)
        assert item['n']%2==1
        assert item['kn']==kappa(item['n'])
    print("data: PASS")
def test_mini_train():
    od='/tmp/collatz_test'
    cfg=C(d=64,nh=4,nel=2,ndl=1,ff=128,msl=12,
          bs=16,ep=2,epe=64,tst=32,ckf=1,nw=0,
          nmax=10**6,dev='cpu',out=od)
    from train import train
    m=train(cfg,32)
    assert os.path.exists(f"{od}/b32/log.json")
    assert os.path.exists(f"{od}/b32/ck_0001.pt")
    assert os.path.exists(f"{od}/b32/best.pt")
    print("mini_train: PASS")
def test_probe():
    od='/tmp/collatz_test'
    cfg=C(d=64,nh=4,nel=2,ndl=1,ff=128,msl=12,
          bs=16,pts=100,pvs=20,kmax=8,pep=5,
          nmax=10**6,dev='cpu',out=od)
    from model import CTF
    m=CTF(cfg,32).to('cpu')
    from probe import collect_acts,train_probe
    from functools import partial
    from data import CFixed,collate
    cf=partial(collate,base=32,msl=12)
    ds=CFixed(100,32,12,10**6,seed=77)
    dl=torch.utils.data.DataLoader(ds,batch_size=16,collate_fn=cf)
    acts,ks,kps=collect_acts(m,dl,cfg)
    assert len(acts)==3
    assert acts[0].shape==(100,64)
    acc=train_probe(acts[0],ks,8,64,ep=5,dev='cpu')
    assert 0<=acc<=1
    print("probe: PASS")
def test_steering():
    cfg=C(d=64,nh=4,nel=2,ndl=1,ff=128,msl=12,
          bs=16,tst=32,sn=50,nmax=10**6,dev='cpu',out='/tmp/collatz_test')
    from model import CTF
    from steer import collect_grouped,compute_sv
    m=CTF(cfg,32).to('cpu')
    grp=collect_grouped(m,cfg,32,ntotal=200)
    assert len(grp)>0
    vk,vkp=compute_sv(grp,cfg.nel,cfg.d,min_samples=2)
    assert 0 in vkp
    assert vkp[0].shape==(64,)
    print("steering: PASS")
def test_evaluate():
    cfg=C(d=64,nh=4,nel=2,ndl=1,ff=128,msl=12,
          bs=16,tst=32,nmax=10**6,dev='cpu',out='/tmp/collatz_test')
    from model import CTF
    from evaluate import eval_acc,analyze_errors,classify_err
    from data import CFixed,collate
    from functools import partial
    m=CTF(cfg,32).to('cpu')
    cf=partial(collate,base=32,msl=12)
    ds=CFixed(32,32,12,10**6,seed=99)
    dl=torch.utils.data.DataLoader(ds,batch_size=16,collate_fn=cf)
    acc=eval_acc(m,dl,32,cfg)
    assert 0<=acc<=1
    errs=analyze_errors(m,dl,32,cfg)
    for e in errs:
        c,v=classify_err(e)
        assert c in['p2','np2','hard','other','neg']
    print("evaluate: PASS")
if __name__=='__main__':
    test_collatz()
    test_enc_dec()
    test_model()
    test_data()
    test_evaluate()
    test_probe()
    test_steering()
    test_mini_train()
    print("\n=== ALL TESTS PASSED ===")
