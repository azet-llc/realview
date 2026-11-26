import glob, os, sys, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from realview.run import default_device
dev = default_device()
# Linear-ish probe: can a 2-layer MLP regress VAE latents from the encoder tokens alone? (run from the repo root)
ps=sorted(glob.glob('data/prep/*.npz')); ho=['2018-08-17--12-07-08','2018-08-10--22-42-26']
tr=[p for p in ps if not any(h in p for h in ho)][:20]; va=[p for p in ps if any(h in p for h in ho)]
def ld(L):
    F,Z=[],[]
    for p in L:
        z=np.load(p); F.append(z['feat'][::4]); Z.append(z['lat'][::4])
    return torch.from_numpy(np.concatenate(F)).float(), torch.from_numpy(np.concatenate(Z)).float()
Ft,Zt=ld(tr); Fv,Zv=ld(va)
m=Zt.mean((0,2,3),keepdim=True); s=Zt.std((0,2,3),keepdim=True); Zt=(Zt-m)/s; Zv=(Zv-m)/s
def tolat(o):  # (N,48,64) -> (N,4,24,32)
    N=o.shape[0]; return o.view(N,6,8,4,4,4).permute(0,3,1,4,2,5).reshape(N,4,24,32)
net=nn.Sequential(nn.LayerNorm(384),nn.Linear(384,512),nn.GELU(),nn.Linear(512,64)).to(dev)
opt=torch.optim.Adam(net.parameters(),1e-3)
Ft,Zt,Fv,Zv=[x.to(dev) for x in (Ft,Zt,Fv,Zv)]
for i in range(1500):
    idx=torch.randint(0,len(Ft),(256,),device=dev); l=(tolat(net(Ft[idx]))-Zt[idx]).pow(2).mean(); opt.zero_grad(); l.backward(); opt.step()
with torch.no_grad(): print('probe val mse',float((tolat(net(Fv))-Zv).pow(2).mean()),'zero',float(Zv.pow(2).mean()))
