import torch,torch.nn as nn,math
class Enc(nn.Module):
    def __init__(self,vs,d,nh,nl,ff,do,msl):
        super().__init__()
        self.emb=nn.Embedding(vs,d)
        self.pe=nn.Embedding(msl,d)
        self.ls=nn.ModuleList([
            nn.TransformerEncoderLayer(d,nh,ff,do,batch_first=True)
            for _ in range(nl)])
        self.ln=nn.LayerNorm(d);self.d=d
    def forward(self,x,mask=None,rh=False,steer=None):
        b,s=x.shape
        p=torch.arange(s,device=x.device).unsqueeze(0)
        h=self.emb(x)*math.sqrt(self.d)+self.pe(p)
        hs=[h]
        km=~mask if mask is not None else None
        for i,l in enumerate(self.ls):
            h=l(h,src_key_padding_mask=km)
            if steer and i in steer:
                h=h+steer[i]
            hs.append(h)
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
class CTF(nn.Module):
    def __init__(self,cfg,base):
        super().__init__()
        vs=base+3
        self.enc=Enc(vs,cfg.d,cfg.nh,cfg.nel,cfg.ff,cfg.do,cfg.msl)
        self.dec=Dec(vs,cfg.d,cfg.nh,cfg.ndl,cfg.ff,cfg.do,cfg.msl)
        self.base=base;self.vs=vs
    def forward(self,src,tgt,sm=None,tm=None,rh=False,steer=None):
        r=self.enc(src,sm,rh=rh,steer=steer)
        if rh:mem,hs=r
        else:mem=r;hs=None
        out=self.dec(tgt,mem,sm,tm)
        return(out,hs)if rh else out
    def generate(self,src,sm=None,ml=16,steer=None):
        r=self.enc(src,sm,steer=steer)
        if isinstance(r,tuple):mem=r[0]
        else:mem=r
        b=src.size(0);dev=src.device;bi=self.base
        ys=torch.full((b,1),bi,dtype=torch.long,device=dev)
        for _ in range(ml-1):
            out=self.dec(ys,mem,sm)
            nx=out[:,-1,:].argmax(-1,keepdim=True)
            ys=torch.cat([ys,nx],1)
            if(nx==bi+1).all():break
        return ys
    def enc_acts(self,src,sm=None,steer=None):
        _,hs=self.enc(src,sm,rh=True,steer=steer)
        return hs
