import json,os,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def plot_learning(logfile,outdir):
    with open(logfile)as f:log=json.load(f)
    eps=[x['ep']for x in log];accs=[x['acc']for x in log]
    losses=[x['loss']for x in log]
    fig,ax1=plt.subplots(figsize=(10,5))
    ax1.plot(eps,accs,'b-',linewidth=1.5)
    ax1.set_xlabel('Epoch (300k examples)');ax1.set_ylabel('Accuracy',color='b')
    ax1.set_ylim(0,1.05);ax1.tick_params(axis='y',labelcolor='b')
    ax2=ax1.twinx()
    ax2.plot(eps,losses,'r-',alpha=0.5,linewidth=0.8)
    ax2.set_ylabel('Loss',color='r');ax2.tick_params(axis='y',labelcolor='r')
    plt.title('Learning Curve');plt.tight_layout()
    plt.savefig(f"{outdir}/learning_curve.png",dpi=150)
    plt.close()
def plot_probes(probefile,logfile,outdir):
    with open(probefile)as f:probes=json.load(f)
    with open(logfile)as f:log=json.load(f)
    acc_by_ep={x['ep']:x['acc']for x in log}
    layers=sorted(set(x['layer']for x in probes))
    epochs=sorted(set(x['ep']for x in probes))
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for l in layers:
        sub=[x for x in probes if x['layer']==l]
        sub.sort(key=lambda x:x['ep'])
        es=[x['ep']for x in sub]
        k_acc=[x['k_lin']for x in sub]
        kp_acc=[x['kp_lin']for x in sub]
        axes[0].plot(es,k_acc,'-o',markersize=3,label=f'L{l}')
        axes[1].plot(es,kp_acc,'-o',markersize=3,label=f'L{l}')
    model_eps=sorted(acc_by_ep.keys())
    model_accs=[acc_by_ep[e]for e in model_eps]
    axes[0].plot(model_eps,model_accs,'k--',linewidth=2,label='model acc',alpha=0.5)
    axes[1].plot(model_eps,model_accs,'k--',linewidth=2,label='model acc',alpha=0.5)
    axes[0].set_title('Probe accuracy for k');axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy');axes[0].legend()
    axes[1].set_title('Probe accuracy for k\'');axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy');axes[1].legend()
    plt.suptitle('Progress Measure: Probe vs Model Accuracy')
    plt.tight_layout();plt.savefig(f"{outdir}/probe_progress.png",dpi=150)
    plt.close()
def plot_error_ratios(errfile,outdir):
    with open(errfile)as f:errs=json.load(f)
    ratios=[e['r']for e in errs if 0<e['r']<100]
    if not ratios:return
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    axes[0].hist(ratios,bins=200,range=(0,10),edgecolor='none')
    axes[0].set_xlabel('p/t ratio');axes[0].set_ylabel('Count')
    axes[0].set_title('All errors')
    for pw in range(1,5):
        axes[0].axvline(2**pw,color='r',linestyle='--',alpha=0.3)
        if 2**(-pw)>0:axes[0].axvline(2**(-pw),color='r',linestyle='--',alpha=0.3)
    lr=[np.log2(r)for r in ratios if r>0]
    axes[1].hist(lr,bins=200,range=(-5,5),edgecolor='none')
    axes[1].set_xlabel('log2(p/t)');axes[1].set_ylabel('Count')
    axes[1].set_title('Log-scale ratios')
    for pw in range(-4,5):
        axes[1].axvline(pw,color='r',linestyle='--',alpha=0.3)
    plt.tight_layout();plt.savefig(f"{outdir}/error_ratios.png",dpi=150)
    plt.close()
def plot_kk_heatmap(kkfile,outdir):
    with open(kkfile)as f:stats=json.load(f)
    kmax=0;kpmax=0
    for key in stats:
        k,kp=map(int,key.split(','))
        kmax=max(kmax,k);kpmax=max(kpmax,kp)
    kmax=min(kmax,12);kpmax=min(kpmax,12)
    grid=np.full((kmax+1,kpmax+1),np.nan)
    for key,v in stats.items():
        k,kp=map(int,key.split(','))
        if k<=kmax and kp<=kpmax:grid[k,kp]=v['acc']
    fig,ax=plt.subplots(figsize=(8,6))
    im=ax.imshow(grid,origin='lower',cmap='RdYlGn',vmin=0,vmax=1,aspect='auto')
    ax.set_xlabel("k'");ax.set_ylabel('k')
    ax.set_title('Accuracy by (k,k\') class')
    plt.colorbar(im,label='Accuracy')
    plt.tight_layout();plt.savefig(f"{outdir}/kk_heatmap.png",dpi=150)
    plt.close()
def plot_steering(steerfile,outdir):
    with open(steerfile)as f:data=json.load(f)
    fig,axes=plt.subplots(1,len(data),figsize=(5*len(data),5))
    if len(data)==1:axes=[axes]
    for idx,(layer,ld)in enumerate(sorted(data.items())):
        ax=axes[idx]
        if'kp'in ld:
            alphas=[];accs=[];fixed=[]
            for a,v in sorted(ld['kp'].items(),key=lambda x:float(x[0])):
                alphas.append(float(a));accs.append(v['acc'])
                fixed.append(v.get('fixed_rate',0))
            ax.plot(alphas,accs,'b-o',label='accuracy')
            ax.plot(alphas,fixed,'g-s',label='fix rate')
        if'random'in ld:
            ra={};rc={}
            for r in ld['random']:
                a=r['alpha']
                ra.setdefault(a,[]).append(r['acc'])
            rmean=[np.mean(ra.get(a,[0]))for a in alphas]if alphas else[]
            if rmean:ax.plot(alphas,rmean,'r--',label='random baseline',alpha=0.5)
        ax.set_xlabel('alpha');ax.set_ylabel('Accuracy')
        ax.set_title(f'Layer {layer}');ax.legend(fontsize=8)
        ax.set_ylim(0,1.05)
    plt.suptitle("k' Steering Results")
    plt.tight_layout();plt.savefig(f"{outdir}/steering.png",dpi=150)
    plt.close()
def plot_all(base,outdir):
    od=f"{outdir}/b{base}"
    if os.path.exists(f"{od}/log.json"):
        plot_learning(f"{od}/log.json",od)
    if os.path.exists(f"{od}/probe_results.json")and os.path.exists(f"{od}/log.json"):
        plot_probes(f"{od}/probe_results.json",f"{od}/log.json",od)
    if os.path.exists(f"{od}/errors.json"):
        plot_error_ratios(f"{od}/errors.json",od)
    if os.path.exists(f"{od}/kk_stats.json"):
        plot_kk_heatmap(f"{od}/kk_stats.json",od)
    if os.path.exists(f"{od}/steer_results.json"):
        plot_steering(f"{od}/steer_results.json",od)
if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--base',type=int,default=32)
    ap.add_argument('--out',type=str,default='output')
    a=ap.parse_args()
    plot_all(a.base,a.out)
