import torch,json,os
from collections import defaultdict
from functools import partial
from data import CFixed,collate,dec as ddec,BOS,EOS,PAD
from config import C
def collect_grouped(m,cfg,base,ntotal=100000):
    cf=partial(collate,base=base,msl=cfg.msl)
    ds=CFixed(ntotal,base,cfg.msl,cfg.nmax,seed=55)
    dl=torch.utils.data.DataLoader(ds,batch_size=cfg.bs,collate_fn=cf)
    grp=defaultdict(lambda:{l:[]for l in range(cfg.nel+1)})
    m.eval()
    with torch.no_grad():
        for bat in dl:
            s=bat['s'].to(cfg.dev);sm=bat['sm'].to(cfg.dev)
            hs=m.enc_acts(s,sm)
            mk=sm.unsqueeze(-1).float()
            denom=mk.sum(1).clamp(min=1)
            for i in range(s.size(0)):
                k=bat['k'][i].item();kp=bat['kp'][i].item()
                for l in range(cfg.nel+1):
                    mki=mk[i]
                    a=(hs[l][i]*mki).sum(0)/mki.sum()
                    grp[(k,kp)][l].append(a.cpu())
    return grp
def compute_sv(grp,nel,d,min_samples=20):
    vkp={};vk={}
    ks=set();kps=set()
    for(k,kp)in grp:ks.add(k);kps.add(kp)
    for l in range(nel+1):
        dkp=[];dk=[]
        for kp in sorted(kps):
            if kp+1 not in kps:continue
            a0=[];a1=[]
            for(k2,kp2),acts in grp.items():
                if kp2==kp and len(acts[l])>=min_samples:
                    a0.extend(acts[l][:min_samples])
                elif kp2==kp+1 and len(acts[l])>=min_samples:
                    a1.extend(acts[l][:min_samples])
            if len(a0)>=min_samples and len(a1)>=min_samples:
                m0=torch.stack(a0).mean(0);m1=torch.stack(a1).mean(0)
                dkp.append(m1-m0)
        for k in sorted(ks):
            if k+1 not in ks:continue
            a0=[];a1=[]
            for(k2,kp2),acts in grp.items():
                if k2==k and len(acts[l])>=min_samples:
                    a0.extend(acts[l][:min_samples])
                elif k2==k+1 and len(acts[l])>=min_samples:
                    a1.extend(acts[l][:min_samples])
            if len(a0)>=min_samples and len(a1)>=min_samples:
                m0=torch.stack(a0).mean(0);m1=torch.stack(a1).mean(0)
                dk.append(m1-m0)
        vkp[l]=torch.stack(dkp).mean(0)if dkp else torch.zeros(d)
        vk[l]=torch.stack(dk).mean(0)if dk else torch.zeros(d)
    return vk,vkp
def _decode_pred(pred,base,i):
    p=pred[i].tolist()
    bi=BOS(base)
    if bi in p:p=p[p.index(bi)+1:]
    ei=EOS(base)
    if ei in p:p=p[:p.index(ei)]
    p=[x for x in p if 0<=x<base]
    return ddec(p,base)if p else -1
def steer_eval(m,cfg,base,vk,vkp,layer,alphas=None):
    if alphas is None:alphas=[-3,-2,-1,-0.5,0,0.5,1,2,3]
    cf=partial(collate,base=base,msl=cfg.msl)
    ds=CFixed(cfg.tst,base,cfg.msl,cfg.nmax,seed=99)
    dl=torch.utils.data.DataLoader(ds,batch_size=cfg.bs,collate_fn=cf)
    m.eval()
    base_preds={}
    with torch.no_grad():
        idx=0
        for bat in dl:
            s=bat['s'].to(cfg.dev);sm=bat['sm'].to(cfg.dev)
            pred=m.generate(s,sm,ml=cfg.msl)
            for i in range(s.size(0)):
                pn=_decode_pred(pred,base,i)
                base_preds[idx]={'p':pn,'t':bat['kn'][i],'k':bat['k'][i].item(),
                                 'kp':bat['kp'][i].item(),'n':bat['n'][i]}
                idx+=1
    results={}
    for a in alphas:
        sv_kp=vkp[layer].to(cfg.dev)*a
        steer={layer:sv_kp.unsqueeze(0).unsqueeze(0)}
        cor=0;tot=0;fixed=0;broke=0
        ratio_shifts=[]
        with torch.no_grad():
            idx=0
            for bat in dl:
                s=bat['s'].to(cfg.dev);sm=bat['sm'].to(cfg.dev)
                pred=m.generate(s,sm,ml=cfg.msl,steer=steer)
                for i in range(s.size(0)):
                    ps=_decode_pred(pred,base,i)
                    t=bat['kn'][i];pb=base_preds[idx]['p']
                    if ps==t:cor+=1
                    if pb!=t and ps==t:fixed+=1
                    if pb==t and ps!=t:broke+=1
                    if pb>0 and ps>0:
                        ratio_shifts.append(ps/pb)
                    tot+=1;idx+=1
        acc=cor/tot;fr=fixed/tot;br=broke/tot
        results[str(a)]={'acc':acc,'fixed_rate':fr,'broke_rate':br,
                         'fixed':fixed,'broke':broke,'total':tot}
        print(f"  a={a:+.1f} acc={acc:.4f} fixed={fixed} broke={broke}")
    return results
def steer_eval_k(m,cfg,base,vk,vkp,layer,alphas=None):
    if alphas is None:alphas=[-3,-2,-1,-0.5,0,0.5,1,2,3]
    cf=partial(collate,base=base,msl=cfg.msl)
    ds=CFixed(cfg.tst,base,cfg.msl,cfg.nmax,seed=99)
    dl=torch.utils.data.DataLoader(ds,batch_size=cfg.bs,collate_fn=cf)
    m.eval()
    results={}
    for a in alphas:
        sv_k=vk[layer].to(cfg.dev)*a
        steer={layer:sv_k.unsqueeze(0).unsqueeze(0)}
        cor=0;tot=0;fixed=0
        with torch.no_grad():
            for bat in dl:
                s=bat['s'].to(cfg.dev);sm=bat['sm'].to(cfg.dev)
                pred_base=m.generate(s,sm,ml=cfg.msl)
                pred_steer=m.generate(s,sm,ml=cfg.msl,steer=steer)
                for i in range(s.size(0)):
                    ps=_decode_pred(pred_steer,base,i)
                    pb=_decode_pred(pred_base,base,i)
                    t=bat['kn'][i]
                    if ps==t:cor+=1
                    if pb!=t and ps==t:fixed+=1
                    tot+=1
        results[str(a)]={'acc':cor/tot,'fixed':fixed,'total':tot}
        print(f"  k-steer a={a:+.1f} acc={cor/tot:.4f} fixed={fixed}")
    return results
def random_baseline(m,cfg,base,layer,n_dirs=5,alphas=None):
    if alphas is None:alphas=[-2,-1,0,1,2]
    cf=partial(collate,base=base,msl=cfg.msl)
    ds=CFixed(min(cfg.tst,10000),base,cfg.msl,cfg.nmax,seed=99)
    dl=torch.utils.data.DataLoader(ds,batch_size=cfg.bs,collate_fn=cf)
    m.eval();results=[]
    for d in range(n_dirs):
        rv=torch.randn(cfg.d);rv=rv/rv.norm()
        for a in alphas:
            sv=rv.to(cfg.dev)*a*10
            steer={layer:sv.unsqueeze(0).unsqueeze(0)}
            cor=0;tot=0
            with torch.no_grad():
                for bat in dl:
                    s=bat['s'].to(cfg.dev);sm=bat['sm'].to(cfg.dev)
                    pred=m.generate(s,sm,ml=cfg.msl,steer=steer)
                    for i in range(s.size(0)):
                        ps=_decode_pred(pred,base,i)
                        if ps==bat['kn'][i]:cor+=1
                        tot+=1
            results.append({'dir':d,'alpha':a,'acc':cor/tot})
    return results
def run_steering(cfg,base,ckpt):
    from model import CTF
    m=CTF(cfg,base).to(cfg.dev)
    sd=torch.load(ckpt,map_location=cfg.dev,weights_only=False)
    m.load_state_dict(sd['model'])
    print("collecting grouped activations...")
    grp=collect_grouped(m,cfg,base,cfg.sn*20)
    print(f"groups: {len(grp)}")
    for(k,kp),acts in sorted(grp.items()):
        print(f"  (k={k},kp={kp}): {len(acts[0])} samples")
    vk,vkp=compute_sv(grp,cfg.nel,cfg.d)
    all_results={}
    for l in range(cfg.nel+1):
        nkp=vkp[l].norm().item();nk=vk[l].norm().item()
        print(f"\n=== Layer {l} (||v_kp||={nkp:.2f}, ||v_k||={nk:.2f}) ===")
        print("kp-steering:")
        rkp=steer_eval(m,cfg,base,vk,vkp,l)
        print("k-steering:")
        rk=steer_eval_k(m,cfg,base,vk,vkp,l)
        print("random baseline:")
        rb=random_baseline(m,cfg,base,l)
        all_results[l]={'kp':rkp,'k':rk,'random':rb,
                        'norms':{'vkp':nkp,'vk':nk}}
    od=f"{cfg.out}/b{base}"
    os.makedirs(od,exist_ok=True)
    with open(f"{od}/steer_results.json",'w')as f:
        json.dump({str(k):v for k,v in all_results.items()},f,default=str)
    torch.save({'vk':vk,'vkp':vkp},f"{od}/steer_vectors.pt")
    return all_results
if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--base',type=int,default=32)
    ap.add_argument('--ckpt',type=str,required=True)
    ap.add_argument('--dev',type=str,default='cuda')
    ap.add_argument('--out',type=str,default='output')
    a=ap.parse_args()
    cfg=C(dev=a.dev,out=a.out)
    run_steering(cfg,a.base,a.ckpt)
