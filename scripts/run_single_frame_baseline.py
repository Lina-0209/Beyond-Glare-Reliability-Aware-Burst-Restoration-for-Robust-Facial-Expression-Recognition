#!/usr/bin/env python3
"""Train once and evaluate the fixed single-frame NAFNet baseline."""

from __future__ import annotations

import argparse,csv,json,sys,time
from pathlib import Path
import lpips,torch,yaml
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from datasets.glare_burst_v3_dataset import CombinedGlareBurstDataset,GlareBurstV3Dataset
from models.baselines import NAFNetSingle
from models.common.metrics import psnr,ssim
from models.common.roi_utils import crop_and_resize


def roots(cfg,key): return {k:ROOT/v for k,v in cfg["dataset"][key].items()}
def evaluate(model,dataset,device,perceptual,batch=4):
    rows=[]; loader=DataLoader(dataset,batch_size=batch,num_workers=2)
    with torch.inference_mode():
        for s in loader:
            anchor=s["burst"][:,0].to(device); target=s["target"].to(device); box=s["roi_bbox"].to(device); pred=model(anchor)
            pr=crop_and_resize(pred,box,128); gt=crop_and_resize(target,box,128)
            fl=perceptual(pred*2-1,target*2-1).flatten(); rl=perceptual(pr*2-1,gt*2-1).flatten()
            for i in range(len(anchor)): rows.append({"full_psnr":psnr(pred[i:i+1],target[i:i+1]),"full_ssim":ssim(pred[i:i+1],target[i:i+1]),"full_lpips":float(fl[i]),"roi_psnr":psnr(pr[i:i+1],gt[i:i+1]),"roi_ssim":ssim(pr[i:i+1],gt[i:i+1]),"roi_lpips":float(rl[i])})
    return {k:sum(r[k] for r in rows)/len(rows) for k in rows[0]}


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--config",type=Path,default=ROOT/"configs/berr/combined.yaml"); ap.add_argument("--output-dir",type=Path,default=ROOT/"outputs/nafnet"); ap.add_argument("--epochs",type=int,default=30); args=ap.parse_args()
    torch.manual_seed(42); cfg=yaml.safe_load(args.config.read_text()); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train=CombinedGlareBurstDataset(roots(cfg,"train_roots")); loader=DataLoader(train,batch_size=8,shuffle=True,num_workers=4,pin_memory=True)
    model=NAFNetSingle().to(device); opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4); scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.epochs)
    args.output_dir.mkdir(parents=True,exist_ok=True); log=[]
    for epoch in range(args.epochs):
        model.train(); total=0
        for s in loader:
            anchor=s["burst"][:,0].to(device); target=s["target"].to(device); box=s["roi_bbox"].to(device); pred=model(anchor)
            loss=torch.nn.functional.l1_loss(pred,target)+torch.nn.functional.l1_loss(crop_and_resize(pred,box,128),crop_and_resize(target,box,128))+.2*(1-ssim(pred,target))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); total+=float(loss)*len(anchor)
        scheduler.step(); row={"epoch":epoch+1,"loss":total/len(train)}; log.append(row); print(row,flush=True)
    checkpoint=args.output_dir/"nafnet_single.pt"; torch.save({"model":model.state_dict(),"epochs":args.epochs,"fixed_configuration":{"width":24,"blocks":[1,1,2,2,1,1,1]}},checkpoint)
    model.eval().requires_grad_(False); perceptual=lpips.LPIPS(net="alex").to(device).eval().requires_grad_(False)
    results={split:{name:evaluate(model,GlareBurstV3Dataset(ROOT/path),device,perceptual) for name,path in cfg["dataset"][f"{split}_roots"].items()} for split in ("val","test") if f"{split}_roots" in cfg["dataset"]}
    sample=train[0]["burst"][0:1].to(device)
    for _ in range(10): model(sample)
    torch.cuda.synchronize() if device.type=="cuda" else None; times=[]
    for _ in range(100):
        start=time.perf_counter(); model(sample); torch.cuda.synchronize() if device.type=="cuda" else None; times.append((time.perf_counter()-start)*1000)
    report={"checkpoint":str(checkpoint),"parameters":sum(p.numel() for p in model.parameters()),"latency_ms":{"mean":sum(times)/len(times),"p50":float(torch.tensor(times).median()),"p95":float(torch.tensor(times).quantile(.95))},"results":results,"training_log":log}
    (args.output_dir/"results.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    with (args.output_dir/"results.csv").open("w",newline="") as f:
        w=csv.writer(f); w.writerow(["split","model","domain","parameters","latency_mean_ms","full_psnr","full_ssim","full_lpips","roi_psnr","roi_ssim","roi_lpips"])
        for split,domains in results.items():
            for domain,m in domains.items(): w.writerow([split,"NAFNetSingle",domain,report["parameters"],report["latency_ms"]["mean"],*[m[k] for k in ("full_psnr","full_ssim","full_lpips","roi_psnr","roi_ssim","roi_lpips")]])
    print(json.dumps({k:v for k,v in report.items() if k!="training_log"},indent=2))

if __name__=="__main__": main()
