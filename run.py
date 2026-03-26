import argparse,os,json,torch
from config import C
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('cmd',choices=['train','eval','probe','steer','transcoder','plots','all'])
    ap.add_argument('--base',type=int,default=32)
    ap.add_argument('--ckpt',type=str,default=None)
    ap.add_argument('--dev',type=str,default='cuda')
    ap.add_argument('--out',type=str,default='output')
    ap.add_argument('--ep',type=int,default=None)
    ap.add_argument('--bs',type=int,default=None)
    ap.add_argument('--nw',type=int,default=None)
    ap.add_argument('--bases',type=str,default=None)
    a=ap.parse_args()
    cfg=C(dev=a.dev,out=a.out)
    if a.ep:cfg.ep=a.ep
    if a.bs:cfg.bs=a.bs
    if a.nw is not None:cfg.nw=a.nw
    if a.bases:cfg.bases=[int(x)for x in a.bases.split(',')]
    if a.cmd=='train':
        from train import train
        train(cfg,a.base)
    elif a.cmd=='eval':
        from model import CTF
        from evaluate import full_error_analysis,eval_per_kk
        from data import CFixed,collate
        from functools import partial
        ckpt=a.ckpt or f"{cfg.out}/b{a.base}/best.pt"
        m=CTF(cfg,a.base).to(cfg.dev)
        sd=torch.load(ckpt,map_location=cfg.dev,weights_only=False)
        m.load_state_dict(sd['model'])
        cf=partial(collate,base=a.base,msl=cfg.msl)
        ds=CFixed(cfg.tst,a.base,cfg.msl,cfg.nmax,seed=99)
        dl=torch.utils.data.DataLoader(ds,batch_size=cfg.bs,collate_fn=cf)
        r=full_error_analysis(m,dl,a.base,cfg)
        print(f"Accuracy: {r['acc']:.4f}")
        print(f"Errors: {r['errors']}/{r['total']}")
        print(f"Categories: {r['cats']}")
        od=f"{cfg.out}/b{a.base}"
        os.makedirs(od,exist_ok=True)
        with open(f"{od}/errors.json",'w')as f:
            json.dump(r['err_list'],f,default=str)
        ds2=CFixed(cfg.tst,a.base,cfg.msl,cfg.nmax,seed=99)
        dl2=torch.utils.data.DataLoader(ds2,batch_size=cfg.bs,collate_fn=cf)
        kk=eval_per_kk(m,dl2,a.base,cfg)
        with open(f"{od}/kk_stats.json",'w')as f:json.dump(kk,f)
        print("\nPer (k,k') accuracy:")
        for key in sorted(kk.keys(),key=lambda x:tuple(map(int,x.split(',')))):
            v=kk[key]
            if v['t']>=10:print(f"  ({key}): {v['acc']:.3f} ({v['c']}/{v['t']})")
    elif a.cmd=='probe':
        from probe import run_probes
        import glob
        if a.ckpt:ckpts=[a.ckpt]
        else:ckpts=sorted(glob.glob(f"{cfg.out}/b{a.base}/ck_*.pt"))
        if not ckpts:print("no checkpoints found");return
        r=run_probes(cfg,a.base,ckpts)
        od=f"{cfg.out}/b{a.base}"
        os.makedirs(od,exist_ok=True)
        with open(f"{od}/probe_results.json",'w')as f:json.dump(r,f)
    elif a.cmd=='steer':
        from steer import run_steering
        ckpt=a.ckpt or f"{cfg.out}/b{a.base}/best.pt"
        run_steering(cfg,a.base,ckpt)
    elif a.cmd=='transcoder':
        from transcoder import train_clt
        ckpt=a.ckpt or f"{cfg.out}/b{a.base}/best.pt"
        clt=train_clt(cfg,a.base,ckpt)
        od=f"{cfg.out}/b{a.base}"
        os.makedirs(od,exist_ok=True)
        torch.save(clt.state_dict(),f"{od}/clt.pt")
    elif a.cmd=='plots':
        from plots import plot_all
        for b in(cfg.bases if not a.base else[a.base]):
            plot_all(b,cfg.out)
    elif a.cmd=='all':
        from train import train
        for b in cfg.bases:
            print(f"\n{'='*60}\n  BASE {b}\n{'='*60}")
            train(cfg,b)
if __name__=='__main__':main()
