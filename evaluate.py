import torch,json
from collections import defaultdict
from data import dec as ddec,BOS,EOS,PAD
def _decode_pred(pred,base,i):
    p=pred[i].tolist()
    bi=BOS(base)
    if bi in p:p=p[p.index(bi)+1:]
    ei=EOS(base)
    if ei in p:p=p[:p.index(ei)]
    p=[x for x in p if 0<=x<base]
    return ddec(p,base)if p else -1
def eval_acc(m,dl,base,cfg):
    m.eval();cor=0;tot=0
    with torch.no_grad():
        for bat in dl:
            s=bat['s'].to(cfg.dev);sm=bat['sm'].to(cfg.dev)
            pred=m.generate(s,sm,ml=cfg.msl)
            for i in range(s.size(0)):
                pn=_decode_pred(pred,base,i)
                if pn==bat['kn'][i]:cor+=1
                tot+=1
    return cor/tot if tot>0 else 0
def eval_per_kk(m,dl,base,cfg):
    m.eval();stats=defaultdict(lambda:{'c':0,'t':0})
    with torch.no_grad():
        for bat in dl:
            s=bat['s'].to(cfg.dev);sm=bat['sm'].to(cfg.dev)
            pred=m.generate(s,sm,ml=cfg.msl)
            for i in range(s.size(0)):
                pn=_decode_pred(pred,base,i)
                k=bat['k'][i].item();kp=bat['kp'][i].item()
                key=f"{k},{kp}"
                stats[key]['t']+=1
                if pn==bat['kn'][i]:stats[key]['c']+=1
    return{k:{'acc':v['c']/v['t']if v['t']>0 else 0,**v}for k,v in stats.items()}
def analyze_errors(m,dl,base,cfg):
    m.eval();errs=[]
    with torch.no_grad():
        for bat in dl:
            s=bat['s'].to(cfg.dev);sm=bat['sm'].to(cfg.dev)
            pred=m.generate(s,sm,ml=cfg.msl)
            for i in range(s.size(0)):
                pn=_decode_pred(pred,base,i)
                t=bat['kn'][i]
                if pn!=t:
                    r=pn/t if t!=0 else float('inf')
                    errs.append({'n':bat['n'][i],'t':t,'p':pn,'r':r,
                                'k':bat['k'][i].item(),'kp':bat['kp'][i].item()})
    return errs
def classify_err(e):
    r=e['r']
    if r<=0:return'neg',None
    for pw in range(-10,11):
        if pw==0:continue
        v=2.0**pw
        if abs(r-v)<0.001*v:return'p2',pw
    for pw in range(-10,11):
        if pw==0:continue
        v=2.0**pw
        if abs(r-v)<0.01*v:return'np2',pw
    for a in range(1,6):
        for l in range(-6,7):
            v=(2/3)**a*2.0**l
            if v>0 and abs(r-v)<0.005*v:return'hard',(a,l)
    return'other',None
def full_error_analysis(m,dl,base,cfg):
    errs=analyze_errors(m,dl,base,cfg)
    cats={'p2':0,'np2':0,'hard':0,'other':0,'neg':0}
    for e in errs:
        c,_=classify_err(e);cats[c]+=1
    tot=0
    for bat in dl:tot+=bat['s'].size(0)
    ne=len(errs)
    return{'total':tot,'errors':ne,'acc':1-ne/tot if tot>0 else 0,
           'cats':cats,'err_list':errs}
