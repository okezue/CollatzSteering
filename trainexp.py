import torch,torch.nn as nn,os,json,time,random,argparse,glob,math
from torch.utils.data import IterableDataset,Dataset,DataLoader
from functools import partial
from dataclasses import dataclass
from modexp import modexp,bitlen,popcount,hamming_dist,enc10,dec10,P
SEP=10;BOS_T=11;EOS_T=12;PAD_T=13;VS=14
@dataclass
class CE:
    d:int=512;nh:int=8;nel:int=4;ndl:int=1;ff:int=2048;do:float=0.1
    msl:int=16;bs:int=256;lr:float=3e-5;ep:int=300;epe:int=300000
    tst:int=50000;ckf:int=10;nw:int=4;seed:int=42
    plr:float=1e-3;pep:int=50;pts:int=50000;pvs:int=10000
    lmax:int=13;hmax:int=13
    dev:str="cuda";out:str="outputexp"
class Enc(nn.Module):
    def __init__(self,vs,d,nh,nl,ff,do,msl):
        super().__init__()
        self.emb=nn.Embedding(vs,d)
        self.pe=nn.Embedding(msl,d)
        self.ls=nn.ModuleList([
            nn.TransformerEncoderLayer(d,nh,ff,do,batch_first=True)
            for _ in range(nl)])
        self.ln=nn.LayerNorm(d);self.d=d
    def forward(self,x,mask=None,rh=False):
        b,s=x.shape
        p=torch.arange(s,device=x.device).unsqueeze(0)
        h=self.emb(x)*math.sqrt(self.d)+self.pe(p)
        hs=[h]
        km=~mask if mask is not None else None
        for l in self.ls:
            h=l(h,src_key_padding_mask=km);hs.append(h)
        h=self.ln(h)
        return(h,hs)if rh else h
class Dec(nn.Module):
    def __init__(self,vs,d,nh,nl,ff,do,msl):
        super().__init__()
        self.emb=nn.Embedding(vs,d)
        self.pe=nn.Embedding(msl,d)
        self.ls=nn.ModuleList([
            nn.TransformerDecoderLayer(d,nh,ff,do,batch_first=True)
            for _ in range(nl)])
        self.ln=nn.LayerNorm(d)
        self.proj=nn.Linear(d,vs);self.d=d
    def forward(self,x,mem,sm=None,tm=None):
        b,s=x.shape
        p=torch.arange(s,device=x.device).unsqueeze(0)
        h=self.emb(x)*math.sqrt(self.d)+self.pe(p)
        cm=torch.triu(torch.ones(s,s,device=x.device,dtype=torch.bool),diagonal=1)
        mkm=~sm if sm is not None else None
        tkm=~tm if tm is not None else None
        for l in self.ls:
            h=l(h,mem,tgt_mask=cm,memory_key_padding_mask=mkm,tgt_key_padding_mask=tkm)
        return self.proj(self.ln(h))
class MEModel(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.enc=Enc(VS,cfg.d,cfg.nh,cfg.nel,cfg.ff,cfg.do,cfg.msl)
        self.dec=Dec(VS,cfg.d,cfg.nh,cfg.ndl,cfg.ff,cfg.do,cfg.msl)
        self.vs=VS
    def forward(self,src,tgt,sm=None,tm=None,rh=False):
        r=self.enc(src,sm,rh=rh)
        if rh:mem,hs=r
        else:mem=r;hs=None
        out=self.dec(tgt,mem,sm,tm)
        return(out,hs)if rh else out
    def generate(self,src,sm=None,ml=16):
        mem=self.enc(src,sm)
        if isinstance(mem,tuple):mem=mem[0]
        b=src.size(0);dev=src.device
        ys=torch.full((b,1),BOS_T,dtype=torch.long,device=dev)
        for _ in range(ml-1):
            out=self.dec(ys,mem,sm)
            nx=out[:,-1,:].argmax(-1,keepdim=True)
            ys=torch.cat([ys,nx],1)
            if(nx==EOS_T).all():break
        return ys
    def enc_acts(self,src,sm=None):
        _,hs=self.enc(src,sm,rh=True)
        return hs
def encode_pair(a,b,msl):
    da=enc10(a);db=enc10(b)
    s=[BOS_T]+da+[SEP]+db+[EOS_T]
    return s[:msl]
def encode_result(r):
    return[BOS_T]+enc10(r)+[EOS_T]
def decode_pred_exp(pred,i):
    p=pred[i].tolist()
    if BOS_T in p:p=p[p.index(BOS_T)+1:]
    if EOS_T in p:p=p[:p.index(EOS_T)]
    p=[x for x in p if 0<=x<=9]
    return dec10(p)if p else -1
class MES(IterableDataset):
    def __init__(self,msl,seed=42):
        self.ml=msl;self.sd=seed
    def __iter__(self):
        wi=torch.utils.data.get_worker_info()
        s=self.sd+(wi.id*7919 if wi else 0)
        rng=random.Random(s)
        while True:
            a=rng.randint(1,P-1);b=rng.randint(1,4096)
            r=modexp(a,b)
            sr=encode_pair(a,b,self.ml)
            tg=encode_result(r)
            yield{'s':sr,'t':tg,'a':a,'b':b,'r':r,
                  'L':bitlen(b),'H':popcount(b)}
class MEF(Dataset):
    def __init__(self,sz,msl,seed=99):
        self.ml=msl;self.data=[]
        rng=random.Random(seed)
        for _ in range(sz):
            a=rng.randint(1,P-1);b=rng.randint(1,4096)
            r=modexp(a,b)
            self.data.append((a,b,r,bitlen(b),popcount(b)))
    def __len__(self):return len(self.data)
    def __getitem__(self,i):
        a,b,r,L,H=self.data[i]
        sr=encode_pair(a,b,self.ml)
        tg=encode_result(r)
        return{'s':sr,'t':tg,'a':a,'b':b,'r':r,'L':L,'H':H}
def collate_exp(batch,msl):
    ss=[x['s'] for x in batch];ts=[x['t'] for x in batch]
    sl=min(max(len(s) for s in ss),msl)
    tl=min(max(len(t) for t in ts),msl)
    sp=torch.full((len(batch),sl),PAD_T,dtype=torch.long)
    tp=torch.full((len(batch),tl),PAD_T,dtype=torch.long)
    for i,(s,t) in enumerate(zip(ss,ts)):
        s=s[:sl];t=t[:tl]
        sp[i,:len(s)]=torch.tensor(s)
        tp[i,:len(t)]=torch.tensor(t)
    return{'s':sp,'t':tp,'sm':sp!=PAD_T,'tm':tp!=PAD_T,
           'a':torch.tensor([x['a'] for x in batch]),
           'b':torch.tensor([x['b'] for x in batch]),
           'r':torch.tensor([x['r'] for x in batch]),
           'L':torch.tensor([x['L'] for x in batch]),
           'H':torch.tensor([x['H'] for x in batch])}
def eval_exp(m,dl,cfg):
    m.eval();cor=0;tot=0
    with torch.no_grad():
        for bat in dl:
            s=bat['s'].to(cfg.dev);sm=bat['sm'].to(cfg.dev)
            pred=m.generate(s,sm,ml=cfg.msl)
            for i in range(s.size(0)):
                pn=decode_pred_exp(pred,i)
                if pn==bat['r'][i].item():cor+=1
                tot+=1
    return cor/tot if tot>0 else 0
def collect_acts_exp(m,dl,cfg):
    m.eval()
    acts={l:[]for l in range(cfg.nel+1)}
    Ls=[];Hs=[]
    with torch.no_grad():
        for bat in dl:
            s=bat['s'].to(cfg.dev);sm=bat['sm'].to(cfg.dev)
            hs=m.enc_acts(s,sm)
            mk=sm.unsqueeze(-1).float()
            denom=mk.sum(1).clamp(min=1)
            for l in range(cfg.nel+1):
                mp=((hs[l]*mk).sum(1)/denom).cpu()
                acts[l].append(mp)
            Ls.append(bat['L']);Hs.append(bat['H'])
    for l in acts:acts[l]=torch.cat(acts[l])
    return acts,torch.cat(Ls),torch.cat(Hs)
def error_analysis(m,dl,cfg):
    m.eval();errs=[]
    with torch.no_grad():
        for bat in dl:
            s=bat['s'].to(cfg.dev);sm=bat['sm'].to(cfg.dev)
            pred=m.generate(s,sm,ml=cfg.msl)
            for i in range(s.size(0)):
                pn=decode_pred_exp(pred,i)
                a=bat['a'][i].item();b=bat['b'][i].item()
                r=bat['r'][i].item()
                if pn!=r:
                    e={'a':a,'b':b,'true':r,'pred':pn,
                       'L':bat['L'][i].item(),'H':bat['H'][i].item()}
                    found=False
                    if 0<=pn<P:
                        for db in range(-3,4):
                            bp=b+db
                            if bp>=1 and db!=0 and modexp(a,bp)==pn:
                                e['b_match']=bp
                                e['b_diff']=db
                                e['hamming']=hamming_dist(b,bp)
                                e['bl_diff']=bitlen(bp)-bitlen(b)
                                found=True;break
                        if not found:
                            for hd in range(1,5):
                                for bit in range(bitlen(b)+1):
                                    bp=b^(1<<bit)
                                    if bp>=1 and modexp(a,bp)==pn:
                                        e['b_match']=bp
                                        e['b_diff']=bp-b
                                        e['hamming']=hamming_dist(b,bp)
                                        e['bl_diff']=bitlen(bp)-bitlen(b)
                                        found=True;break
                                if found:break
                    e['matched']=found
                    errs.append(e)
    return errs
def train_exp(cfg):
    od=cfg.out
    os.makedirs(od,exist_ok=True)
    m=MEModel(cfg).to(cfg.dev)
    np_=sum(p.numel() for p in m.parameters())
    print(f"modexp p={P} params={np_:,}")
    opt=torch.optim.Adam(m.parameters(),lr=cfg.lr)
    cf=partial(collate_exp,msl=cfg.msl)
    ds=MES(cfg.msl,cfg.seed)
    dl=DataLoader(ds,batch_size=cfg.bs,collate_fn=cf,num_workers=cfg.nw,
                  pin_memory=(cfg.dev!='cpu'))
    tds=MEF(cfg.tst,cfg.msl,seed=99)
    tdl=DataLoader(tds,batch_size=cfg.bs,collate_fn=cf,
                   pin_memory=(cfg.dev!='cpu'))
    ce=nn.CrossEntropyLoss(ignore_index=PAD_T)
    spe=cfg.epe//cfg.bs
    log=[];best_acc=-1
    it=iter(dl);step=0;t0=time.time()
    for ep in range(1,cfg.ep+1):
        m.train();el=0;ec=0
        for _ in range(spe):
            bat=next(it)
            s=bat['s'].to(cfg.dev);t=bat['t'].to(cfg.dev)
            sm=bat['sm'].to(cfg.dev);tm=bat['tm'].to(cfg.dev)
            out=m(s,t[:,:-1],sm,tm[:,:-1])
            loss=ce(out.reshape(-1,m.vs),t[:,1:].reshape(-1))
            opt.zero_grad();loss.backward();opt.step()
            el+=loss.item();ec+=1;step+=1
        al=el/ec
        acc=eval_exp(m,tdl,cfg)
        dt=time.time()-t0
        log.append({'ep':ep,'loss':al,'acc':acc,'step':step,'time':dt})
        print(f"ep={ep:4d} loss={al:.4f} acc={acc:.4f} t={dt:.0f}s")
        if acc>best_acc:
            best_acc=acc
            torch.save({'model':m.state_dict(),'ep':ep,'acc':acc},
                       f"{od}/best.pt")
        if ep%cfg.ckf==0:
            torch.save({'model':m.state_dict(),'opt':opt.state_dict(),
                        'ep':ep,'acc':acc,'cfg':vars(cfg)},
                       f"{od}/ck_{ep:04d}.pt")
        with open(f"{od}/log.json",'w')as f:json.dump(log,f)
    return m
def run_probes_exp(cfg):
    od=cfg.out
    ckpts=sorted(glob.glob(f"{od}/ck_*.pt"))
    if not ckpts:print("no checkpoints");return
    cf=partial(collate_exp,msl=cfg.msl)
    pds=MEF(cfg.pts+cfg.pvs,cfg.msl,seed=77)
    pdl=DataLoader(pds,batch_size=cfg.bs,collate_fn=cf)
    from probe import train_probe
    results=[]
    for cp in ckpts:
        m=MEModel(cfg).to(cfg.dev)
        sd=torch.load(cp,map_location=cfg.dev,weights_only=False)
        m.load_state_dict(sd['model'])
        ep=sd['ep']
        acts,Ls,Hs=collect_acts_exp(m,pdl,cfg)
        for l in range(cfg.nel+1):
            la_lin=train_probe(acts[l],Ls,cfg.lmax,cfg.d,cfg.plr,cfg.pep,cfg.dev)
            ha_lin=train_probe(acts[l],Hs,cfg.hmax,cfg.d,cfg.plr,cfg.pep,cfg.dev)
            la_mlp=train_probe(acts[l],Ls,cfg.lmax,cfg.d,cfg.plr,cfg.pep,cfg.dev,mlp=True)
            ha_mlp=train_probe(acts[l],Hs,cfg.hmax,cfg.d,cfg.plr,cfg.pep,cfg.dev,mlp=True)
            r={'ep':ep,'layer':l,
               'L_lin':la_lin,'H_lin':ha_lin,
               'L_mlp':la_mlp,'H_mlp':ha_mlp}
            results.append(r)
            print(f"ep={ep} L={l} L_lin={la_lin:.4f} H_lin={ha_lin:.4f} "
                  f"L_mlp={la_mlp:.4f} H_mlp={ha_mlp:.4f}")
        del m;torch.cuda.empty_cache()if cfg.dev=='cuda' else None
    with open(f"{od}/probe_results.json",'w')as f:json.dump(results,f)
    return results
def run_error_analysis(cfg):
    od=cfg.out
    best=f"{od}/best.pt"
    if not os.path.exists(best):print("no best.pt");return
    m=MEModel(cfg).to(cfg.dev)
    sd=torch.load(best,map_location=cfg.dev,weights_only=False)
    m.load_state_dict(sd['model'])
    cf=partial(collate_exp,msl=cfg.msl)
    tds=MEF(cfg.tst,cfg.msl,seed=99)
    tdl=DataLoader(tds,batch_size=cfg.bs,collate_fn=cf)
    errs=error_analysis(m,tdl,cfg)
    n_matched=sum(1 for e in errs if e.get('matched'))
    summary={'total_errors':len(errs),'matched_b':n_matched,
             'match_rate':n_matched/len(errs)if errs else 0}
    if n_matched>0:
        hds=[e['hamming'] for e in errs if e.get('matched')]
        blds=[e['bl_diff'] for e in errs if e.get('matched')]
        summary['avg_hamming']=sum(hds)/len(hds)
        summary['avg_bl_diff']=sum(blds)/len(blds)
    out={'summary':summary,'errors':errs[:1000]}
    with open(f"{od}/errors.json",'w')as f:json.dump(out,f)
    print(f"errors: {len(errs)} total, {n_matched} matched to b'")
    return out
def test_modexp():
    assert modexp(2,10)==pow(2,10,P)
    assert modexp(5,997)==pow(5,997,P)
    for a in[2,3,100,500,996]:
        for b in[1,2,10,100,1000,4096]:
            assert modexp(a,b)==pow(a,b,P)
    assert bitlen(1)==1;assert bitlen(4096)==13
    assert popcount(7)==3;assert popcount(255)==8
    assert hamming_dist(0b1010,0b1001)==2
    s=encode_pair(123,456,16)
    assert BOS_T==s[0]
    assert SEP in s
    assert EOS_T==s[-1]
    r=encode_result(500)
    assert r[0]==BOS_T and r[-1]==EOS_T
    print("test_modexp math: PASS")
    od='/tmp/testexp'
    cfg=CE(d=64,nh=4,nel=2,ndl=1,ff=128,msl=16,
           bs=16,ep=2,epe=64,tst=32,ckf=1,nw=0,
           dev='cpu',out=od,lmax=6,hmax=6,
           pts=50,pvs=10,pep=3)
    m=train_exp(cfg)
    assert os.path.exists(f"{od}/log.json")
    assert os.path.exists(f"{od}/ck_0001.pt")
    assert os.path.exists(f"{od}/best.pt")
    print("test_modexp train: PASS")
    cfg2=CE(d=64,nh=4,nel=2,ndl=1,ff=128,msl=16,
            bs=16,nw=0,dev='cpu',out=od,
            lmax=6,hmax=6,pts=50,pvs=10,pep=3)
    r=run_probes_exp(cfg2)
    assert len(r)>0
    assert os.path.exists(f"{od}/probe_results.json")
    print("test_modexp probes: PASS")
    ea=run_error_analysis(cfg2)
    assert os.path.exists(f"{od}/errors.json")
    print("test_modexp errors: PASS")
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--dev',type=str,default='cuda')
    ap.add_argument('--out',type=str,default='outputexp')
    ap.add_argument('--ep',type=int,default=None)
    ap.add_argument('--bs',type=int,default=None)
    ap.add_argument('--nw',type=int,default=None)
    ap.add_argument('--test',action='store_true')
    a=ap.parse_args()
    if a.test:
        test_modexp()
    else:
        cfg=CE(dev=a.dev,out=a.out)
        if a.ep:cfg.ep=a.ep
        if a.bs:cfg.bs=a.bs
        if a.nw is not None:cfg.nw=a.nw
        m=train_exp(cfg)
        print("running probes...")
        run_probes_exp(cfg)
        print("running error analysis...")
        run_error_analysis(cfg)
