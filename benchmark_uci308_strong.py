#!/usr/bin/env python3
"""
Strong real-data benchmark for UCI 308 Gas Sensor Array under Flow Modulation.

Task: 4-class classification using the official `gas` label
      (air / acetone / ethanol / mixture).
Input: first 13 respiration cycles x 32 channels
       (16 MOX sensors x low/high-frequency cycle features).

Models, all evaluated on identical splits:
  - SVM
  - Random Forest
  - XGBoost
  - LSTM
  - BiLSTM
  - Attention-BiLSTM (Ours)
  - CNN-LSTM

Protocols:
  A) 5x repeated stratified 5-fold CV (25 matched folds)
  B) Leave-one-batch-out (5 natural day/session groups)

Metrics:
  accuracy, macro-F1, balanced accuracy, macro OVR AUC,
  NLL, multiclass Brier, ECE.

Statistics:
  - Friedman test on repeat-level macro-F1
  - paired Wilcoxon on repeat-level macro-F1 (Ours vs baselines)
  - exact McNemar on first-repeat OOF correctness
  - stratified paired permutation test on first-repeat OOF macro-F1

The script downloads the public UCI archive directly and writes all numerical
outputs, figures, an abstract, and a drop-in IEEE Results section to ./results.
"""

from __future__ import annotations
import io, json, math, os, random, re, time, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

from scipy.stats import friedmanchisquare, wilcoxon, binomtest
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score,
    log_loss, confusion_matrix
)
from sklearn.model_selection import RepeatedStratifiedKFold, LeaveOneGroupOut, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ------------------------------ config ---------------------------------
SEED = 2026
N_REPEATS = 5
N_SPLITS = 5
EPOCHS = 180
PATIENCE = 28
BATCH_SIZE = 16
LR = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN = 32
DEVICE = torch.device("cpu")
OUT = Path("results")
OUT.mkdir(parents=True, exist_ok=True)

UCI_URLS = [
    "https://cdn.uci-ics-mlr-prod.aws.uci.edu/308/gas%2Bsensor%2Barray%2Bunder%2Bflow%2Bmodulation.zip",
    "https://archive.ics.uci.edu/static/public/308/gas%2Bsensor%2Barray%2Bunder%2Bflow%2Bmodulation.zip",
]

MODEL_ORDER = ["SVM", "RF", "XGBoost", "LSTM", "BiLSTM", "CNN-LSTM", "Attention-BiLSTM (Ours)"]
DEEP_MODELS = {"LSTM", "BiLSTM", "CNN-LSTM", "Attention-BiLSTM (Ours)"}


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)
    torch.set_num_threads(2)


def download_features() -> pd.DataFrame:
    last_err = None
    for url in UCI_URLS:
        try:
            r = requests.get(url, timeout=90)
            r.raise_for_status()
            z = zipfile.ZipFile(io.BytesIO(r.content))
            names = z.namelist()
            cand = [n for n in names if n.lower().endswith("features.csv")]
            if not cand:
                raise RuntimeError(f"features.csv not found in archive; files={names}")
            with z.open(cand[0]) as f:
                df = pd.read_csv(f)
            (OUT / "uci_download_source.txt").write_text(url + "\n", encoding="utf-8")
            return df
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Unable to download UCI 308: {last_err}")


def build_tensor(df: pd.DataFrame):
    blocks = []
    for cycle in range(1, 14):
        cols = []
        for sensor in range(1, 17):
            cols.extend([f"S{sensor}_r{cycle}_Alf", f"S{sensor}_r{cycle}_Ahf"])
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"Missing columns: {missing[:5]}")
        blocks.append(df[cols].to_numpy(np.float32))
    X = np.stack(blocks, axis=1)  # N x 13 x 32
    le = LabelEncoder()
    y = le.fit_transform(df["gas"].astype(str).to_numpy())
    groups = df["batch"].astype(str).to_numpy()
    meta = df[["exp", "batch", "gas", "lab", "ace_conc", "eth_conc"]].copy()
    return X, y, groups, le, meta


def ece_toplabel(y, p, n_bins=8):
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    corr = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.any():
            acc = corr[mask].mean(); c = conf[mask].mean(); n = int(mask.sum())
            ece += (n / len(y)) * abs(acc - c)
            rows.append((i, lo, hi, n, c, acc))
    return float(ece), rows


def brier_multiclass(y, p, n_classes):
    oh = np.eye(n_classes)[y]
    return float(np.mean(np.sum((p - oh) ** 2, axis=1)))


def metrics(y, p, n_classes):
    pred = p.argmax(axis=1)
    out = {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, labels=np.arange(n_classes), average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "nll": log_loss(y, p, labels=np.arange(n_classes)),
        "brier": brier_multiclass(y, p, n_classes),
        "ece": ece_toplabel(y, p)[0],
    }
    try:
        out["macro_auc_ovr"] = roc_auc_score(y, p, multi_class="ovr", average="macro", labels=np.arange(n_classes))
    except Exception:
        out["macro_auc_ovr"] = np.nan
    return out


class SeqClassifier(nn.Module):
    def __init__(self, kind: str, in_dim: int, n_classes: int):
        super().__init__()
        self.kind = kind
        self.dropout = nn.Dropout(0.25)
        if kind == "LSTM":
            self.rnn = nn.LSTM(in_dim, HIDDEN, batch_first=True)
            self.rep_dim = HIDDEN
        elif kind == "BiLSTM":
            self.rnn = nn.LSTM(in_dim, HIDDEN, batch_first=True, bidirectional=True)
            self.rep_dim = 2 * HIDDEN
        elif kind == "Attention-BiLSTM (Ours)":
            self.rnn = nn.LSTM(in_dim, HIDDEN, batch_first=True, bidirectional=True)
            self.rep_dim = 2 * HIDDEN
            self.attn = nn.Sequential(nn.Linear(self.rep_dim, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, 1))
        elif kind == "CNN-LSTM":
            self.conv = nn.Sequential(
                nn.Conv1d(in_dim, 48, kernel_size=3, padding=1),
                nn.BatchNorm1d(48), nn.ReLU(), nn.Dropout(0.15)
            )
            self.rnn = nn.LSTM(48, HIDDEN, batch_first=True)
            self.rep_dim = HIDDEN
        else:
            raise ValueError(kind)
        self.head = nn.Sequential(nn.Linear(self.rep_dim, 32), nn.ReLU(), nn.Dropout(0.25), nn.Linear(32, n_classes))

    def forward(self, x, return_attention=False):
        if self.kind == "CNN-LSTM":
            x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        out, _ = self.rnn(x)
        att = None
        if self.kind == "Attention-BiLSTM (Ours)":
            s = self.attn(out).squeeze(-1)
            att = torch.softmax(s, dim=1)
            rep = torch.sum(out * att.unsqueeze(-1), dim=1)
        else:
            rep = out.mean(dim=1)
        logits = self.head(self.dropout(rep))
        if return_attention:
            return logits, att
        return logits


def standardize_seq(Xtr, Xte):
    mu = Xtr.mean(axis=(0,1), keepdims=True)
    sd = Xtr.std(axis=(0,1), keepdims=True)
    sd[sd < 1e-7] = 1.0
    return (Xtr-mu)/sd, (Xte-mu)/sd, mu, sd


def train_deep(kind, Xtr, ytr, Xte, seed):
    seed_all(seed)
    n_classes = int(np.max(ytr)) + 1
    Xtr_s, Xte_s, mu, sd = standardize_seq(Xtr, Xte)

    # Internal validation split is used only for early stopping; test fold is untouched.
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.22, random_state=seed)
    itr, iva = next(sss.split(Xtr_s, ytr))
    Xfit, yfit = Xtr_s[itr], ytr[itr]
    Xval, yval = Xtr_s[iva], ytr[iva]

    counts = np.bincount(yfit, minlength=n_classes).astype(float)
    w = counts.sum() / np.maximum(counts, 1.0)
    w = w / w.mean()

    model = SeqClassifier(kind, Xtr.shape[-1], n_classes).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=DEVICE))

    train_ds = TensorDataset(torch.tensor(Xfit, dtype=torch.float32), torch.tensor(yfit, dtype=torch.long))
    loader = DataLoader(train_ds, batch_size=min(BATCH_SIZE, len(train_ds)), shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    Xv = torch.tensor(Xval, dtype=torch.float32, device=DEVICE)
    yv = torch.tensor(yval, dtype=torch.long, device=DEVICE)

    best = None; best_loss = float("inf"); wait = 0
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            xb=xb.to(DEVICE); yb=yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = loss_fn(model(Xv), yv).item()
        if vloss < best_loss - 1e-5:
            best_loss = vloss; wait = 0
            best = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    if best is not None:
        model.load_state_dict(best)
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(Xte_s, dtype=torch.float32, device=DEVICE)
        logits, att = model(Xt, return_attention=True)
        prob = torch.softmax(logits, dim=1).cpu().numpy()
        att_np = None if att is None else att.cpu().numpy()
    return prob, att_np, epoch + 1


def classical_model(name, seed, n_classes):
    if name == "SVM":
        return Pipeline([
            ("sc", StandardScaler()),
            ("clf", SVC(C=10.0, kernel="rbf", gamma="scale", probability=True,
                        class_weight="balanced", random_state=seed))
        ])
    if name == "RF":
        return RandomForestClassifier(
            n_estimators=700, max_features="sqrt", min_samples_leaf=1,
            class_weight="balanced_subsample", random_state=seed, n_jobs=2
        )
    if name == "XGBoost":
        return XGBClassifier(
            objective="multi:softprob", num_class=n_classes,
            n_estimators=350, max_depth=3, learning_rate=0.035,
            subsample=0.85, colsample_bytree=0.80,
            reg_lambda=2.0, min_child_weight=1.0,
            eval_metric="mlogloss", random_state=seed, n_jobs=2
        )
    raise ValueError(name)


def fit_predict(name, Xtr, ytr, Xte, seed, n_classes):
    if name in DEEP_MODELS:
        p, att, epochs = train_deep(name, Xtr, ytr, Xte, seed)
        return p, att, epochs
    flat_tr = Xtr.reshape(len(Xtr), -1)
    flat_te = Xte.reshape(len(Xte), -1)
    m = classical_model(name, seed, n_classes)
    m.fit(flat_tr, ytr)
    p = m.predict_proba(flat_te)
    # sklearn can omit a class only if absent in training; stratification/groups should avoid it,
    # but align defensively to all classes if needed.
    if p.shape[1] != n_classes:
        full = np.full((len(Xte), n_classes), 1e-9)
        full[:, m.classes_.astype(int)] = p
        p = full / full.sum(axis=1, keepdims=True)
    return p, None, np.nan


def run_repeated_cv(X, y, n_classes):
    splitter = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    splits = list(splitter.split(X, y))
    fold_rows=[]; pred_rows=[]; att_rows=[]
    for fold_idx, (tr, te) in enumerate(splits):
        repeat = fold_idx // N_SPLITS
        fold = fold_idx % N_SPLITS
        print(f"CV repeat {repeat+1}/{N_REPEATS} fold {fold+1}/{N_SPLITS}", flush=True)
        for mi, name in enumerate(MODEL_ORDER):
            t0=time.time()
            p, att, epochs = fit_predict(name, X[tr], y[tr], X[te], SEED + 1000*repeat + 100*fold + mi, n_classes)
            met=metrics(y[te],p,n_classes)
            fold_rows.append({"protocol":"repeated_stratified_5x5","repeat":repeat,"fold":fold,
                              "model":name,"n_train":len(tr),"n_test":len(te),"train_seconds":time.time()-t0,
                              "epochs":epochs,**met})
            for loc, idx in enumerate(te):
                row={"repeat":repeat,"fold":fold,"model":name,"sample_idx":int(idx),"y_true":int(y[idx]),
                     "y_pred":int(p[loc].argmax())}
                for c in range(n_classes): row[f"p{c}"]=float(p[loc,c])
                pred_rows.append(row)
                if repeat==0 and name=="Attention-BiLSTM (Ours)" and att is not None:
                    for cyc,a in enumerate(att[loc],1):
                        att_rows.append({"sample_idx":int(idx),"cycle":cyc,"attention":float(a)})
    folds=pd.DataFrame(fold_rows); preds=pd.DataFrame(pred_rows)
    folds.to_csv(OUT/"cv_fold_metrics.csv",index=False)
    preds.to_csv(OUT/"cv_oof_predictions.csv",index=False)
    pd.DataFrame(att_rows).to_csv(OUT/"ours_attention_repeat0.csv",index=False)

    # repeat-level OOF metrics: each sample occurs exactly once per model per repeat
    reps=[]
    for repeat in range(N_REPEATS):
        for name in MODEL_ORDER:
            q=preds[(preds.repeat==repeat)&(preds.model==name)].sort_values("sample_idx")
            yy=q.y_true.to_numpy(int); pp=q[[f"p{i}" for i in range(n_classes)]].to_numpy(float)
            reps.append({"repeat":repeat,"model":name,**metrics(yy,pp,n_classes)})
    repdf=pd.DataFrame(reps); repdf.to_csv(OUT/"cv_repeat_metrics.csv",index=False)
    return folds,preds,repdf


def run_logo(X,y,groups,n_classes):
    logo=LeaveOneGroupOut(); rows=[]; preds=[]
    for fi,(tr,te) in enumerate(logo.split(X,y,groups)):
        batch=str(groups[te][0])
        print(f"LOGO held-out batch {batch}",flush=True)
        for mi,name in enumerate(MODEL_ORDER):
            t0=time.time()
            p,_,epochs=fit_predict(name,X[tr],y[tr],X[te],SEED+50000+fi*100+mi,n_classes)
            met=metrics(y[te],p,n_classes)
            rows.append({"protocol":"leave_one_batch_out","fold":fi,"held_out_batch":batch,"model":name,
                         "n_train":len(tr),"n_test":len(te),"train_seconds":time.time()-t0,"epochs":epochs,**met})
            for loc,idx in enumerate(te):
                rr={"held_out_batch":batch,"model":name,"sample_idx":int(idx),"y_true":int(y[idx]),"y_pred":int(p[loc].argmax())}
                for c in range(n_classes): rr[f"p{c}"]=float(p[loc,c])
                preds.append(rr)
    df=pd.DataFrame(rows); pr=pd.DataFrame(preds)
    df.to_csv(OUT/"batch_holdout_metrics.csv",index=False)
    pr.to_csv(OUT/"batch_holdout_predictions.csv",index=False)
    return df,pr


def holm(pvals):
    items=sorted(pvals.items(), key=lambda kv: kv[1])
    m=len(items); out={}
    running=0.0
    for i,(k,p) in enumerate(items):
        adj=min(1.0,(m-i)*p)
        running=max(running,adj)
        out[k]=running
    return out


def mcnemar_exact(y, pred_a, pred_b):
    ca=(pred_a==y); cb=(pred_b==y)
    b=int(np.sum(ca & ~cb)); c=int(np.sum(~ca & cb))
    n=b+c
    p=1.0 if n==0 else binomtest(min(b,c),n,0.5,alternative="two-sided").pvalue
    return b,c,float(p)


def paired_perm_macro_f1(y, pa, pb, n_classes, n_perm=10000, seed=SEED):
    obs=f1_score(y,pa,labels=np.arange(n_classes),average="macro",zero_division=0)-f1_score(y,pb,labels=np.arange(n_classes),average="macro",zero_division=0)
    rng=np.random.default_rng(seed); ge=0
    for _ in range(n_perm):
        sw=rng.random(len(y))<0.5
        aa=pa.copy(); bb=pb.copy()
        aa[sw],bb[sw]=pb[sw],pa[sw]
        d=f1_score(y,aa,labels=np.arange(n_classes),average="macro",zero_division=0)-f1_score(y,bb,labels=np.arange(n_classes),average="macro",zero_division=0)
        if d>=obs-1e-12: ge+=1
    return float(obs),float((ge+1)/(n_perm+1))


def stats_tests(repdf,preds,n_classes):
    pivot=repdf.pivot(index="repeat",columns="model",values="macro_f1")[MODEL_ORDER]
    fr=friedmanchisquare(*[pivot[m].values for m in MODEL_ORDER])
    rows=[]; pvals={}
    ours=pivot["Attention-BiLSTM (Ours)"].values
    for name in MODEL_ORDER:
        if name=="Attention-BiLSTM (Ours)": continue
        x=pivot[name].values
        try:
            w=wilcoxon(ours,x,alternative="two-sided",method="auto")
            wp=float(w.pvalue); stat=float(w.statistic)
        except Exception:
            wp=1.0; stat=np.nan
        pvals[name]=wp
        rows.append({"comparison":f"Ours vs {name}","baseline":name,"mean_delta_macro_f1":float(np.mean(ours-x)),
                     "median_delta_macro_f1":float(np.median(ours-x)),"wilcoxon_stat":stat,"wilcoxon_p":wp})
    adj=holm(pvals)
    for r in rows:r["wilcoxon_holm_p"]=adj[r["baseline"]]

    # first-repeat OOF paired correctness + paired permutation macro-F1
    first=preds[preds.repeat==0]
    q_ours=first[first.model=="Attention-BiLSTM (Ours)"].sort_values("sample_idx")
    y0=q_ours.y_true.to_numpy(int); po=q_ours.y_pred.to_numpy(int)
    for r in rows:
        name=r["baseline"]
        qb=first[first.model==name].sort_values("sample_idx")
        pb=qb.y_pred.to_numpy(int)
        b,c,pm=mcnemar_exact(y0,po,pb)
        d,pp=paired_perm_macro_f1(y0,po,pb,n_classes,seed=SEED+len(name))
        r.update({"mcnemar_ours_correct_baseline_wrong":b,"mcnemar_ours_wrong_baseline_correct":c,
                  "mcnemar_exact_p":pm,"oof_delta_macro_f1":d,"paired_perm_p_one_sided":pp})
    out=pd.DataFrame(rows); out.to_csv(OUT/"paired_statistics.csv",index=False)
    summary={"friedman_statistic":float(fr.statistic),"friedman_p":float(fr.pvalue),"n_repeats":N_REPEATS,
             "note":"Wilcoxon uses repeat-level OOF macro-F1; McNemar/permutation use first-repeat paired OOF predictions."}
    (OUT/"global_statistics.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    return out,summary


def stratified_bootstrap_metric(y,p,n_classes,metric_name="macro_f1",B=4000,seed=SEED):
    rng=np.random.default_rng(seed)
    cls=[np.where(y==c)[0] for c in range(n_classes)]
    vals=[]
    for _ in range(B):
        idx=np.concatenate([rng.choice(ix,size=len(ix),replace=True) for ix in cls])
        yy=y[idx]; pp=p[idx]
        vals.append(metrics(yy,pp,n_classes)[metric_name])
    return np.percentile(vals,[2.5,97.5]).tolist()


def make_summaries(repdf,preds,batchdf,n_classes,classes):
    rows=[]
    for name in MODEL_ORDER:
        r=repdf[repdf.model==name]
        first=preds[(preds.repeat==0)&(preds.model==name)].sort_values("sample_idx")
        yy=first.y_true.to_numpy(int); pp=first[[f"p{i}" for i in range(n_classes)]].to_numpy(float)
        ci=stratified_bootstrap_metric(yy,pp,n_classes,"macro_f1",B=4000,seed=SEED+MODEL_ORDER.index(name))
        b=batchdf[batchdf.model==name]
        rows.append({
            "model":name,
            "macro_f1_mean":r.macro_f1.mean(),"macro_f1_sd":r.macro_f1.std(ddof=1),
            "macro_f1_repeat0":metrics(yy,pp,n_classes)["macro_f1"],
            "macro_f1_repeat0_boot95_low":ci[0],"macro_f1_repeat0_boot95_high":ci[1],
            "balanced_accuracy_mean":r.balanced_accuracy.mean(),"macro_auc_mean":r.macro_auc_ovr.mean(),
            "ece_mean":r.ece.mean(),"nll_mean":r.nll.mean(),"brier_mean":r.brier.mean(),
            "batch_macro_f1_mean":b.macro_f1.mean(),"batch_macro_f1_sd":b.macro_f1.std(ddof=1),
            "batch_macro_f1_worst":b.macro_f1.min(),"batch_balanced_accuracy_mean":b.balanced_accuracy.mean(),
            "robustness_gap_f1":r.macro_f1.mean()-b.macro_f1.mean(),
            "train_seconds_fold_mean":float(pd.read_csv(OUT/"cv_fold_metrics.csv").query("model==@name").train_seconds.mean())
        })
    s=pd.DataFrame(rows); s.to_csv(OUT/"model_summary.csv",index=False)

    # Confusion and reliability data for first repeat
    for name in MODEL_ORDER:
        q=preds[(preds.repeat==0)&(preds.model==name)].sort_values("sample_idx")
        yy=q.y_true.to_numpy(int); pred=q.y_pred.to_numpy(int)
        cm=confusion_matrix(yy,pred,labels=np.arange(n_classes),normalize="true")
        pd.DataFrame(cm,index=classes,columns=classes).to_csv(OUT/("confusion_"+re.sub(r"[^A-Za-z0-9]+","_",name)+".csv"))
    return s


def figures(summary,preds,batchdf,stats,n_classes,classes):
    # 1. performance-calibration frontier
    fig,ax=plt.subplots(figsize=(7.2,5.2))
    for _,r in summary.iterrows():
        ax.scatter(r.ece_mean,r.macro_f1_mean,s=65)
        ax.annotate(r.model,(r.ece_mean,r.macro_f1_mean),xytext=(5,4),textcoords="offset points",fontsize=8)
    ax.set_xlabel("Expected calibration error (lower is better)")
    ax.set_ylabel("Macro-F1 (higher is better)")
    ax.set_title("Discrimination–calibration frontier on real UCI 308 data")
    ax.grid(True,alpha=.25); fig.tight_layout()
    fig.savefig(OUT/"fig1_performance_calibration_frontier.pdf",bbox_inches="tight")
    fig.savefig(OUT/"fig1_performance_calibration_frontier.png",dpi=300,bbox_inches="tight"); plt.close(fig)

    # 2. cross-batch robustness trajectories
    fig,ax=plt.subplots(figsize=(7.5,5.2))
    batches=list(pd.unique(batchdf.held_out_batch))
    for name in MODEL_ORDER:
        q=batchdf[batchdf.model==name].set_index("held_out_batch").reindex(batches)
        ax.plot(np.arange(len(batches)),q.macro_f1.values,marker="o",linewidth=1.35,label=name)
    ax.set_xticks(np.arange(len(batches)),batches,rotation=25,ha="right")
    ax.set_ylim(0,1.02); ax.set_ylabel("Macro-F1 on unseen batch")
    ax.set_title("Leave-one-batch-out robustness: natural session/domain shift")
    ax.grid(True,alpha=.25); ax.legend(fontsize=7,frameon=False,ncol=2)
    fig.tight_layout(); fig.savefig(OUT/"fig2_batch_robustness.pdf",bbox_inches="tight")
    fig.savefig(OUT/"fig2_batch_robustness.png",dpi=300,bbox_inches="tight"); plt.close(fig)

    # 3. ours normalized confusion matrix, first repeat OOF
    name="Attention-BiLSTM (Ours)"
    q=preds[(preds.repeat==0)&(preds.model==name)].sort_values("sample_idx")
    cm=confusion_matrix(q.y_true,q.y_pred,labels=np.arange(n_classes),normalize="true")
    fig,ax=plt.subplots(figsize=(5.8,5.0))
    im=ax.imshow(cm,vmin=0,vmax=1,aspect="auto")
    for i in range(n_classes):
        for j in range(n_classes): ax.text(j,i,f"{cm[i,j]:.2f}",ha="center",va="center",fontsize=9)
    ax.set_xticks(np.arange(n_classes),classes,rotation=30,ha="right")
    ax.set_yticks(np.arange(n_classes),classes)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    ax.set_title("Attention-BiLSTM OOF normalized confusion matrix")
    fig.colorbar(im,ax=ax,label="Row-normalized fraction")
    fig.tight_layout(); fig.savefig(OUT/"fig3_ours_confusion.pdf",bbox_inches="tight")
    fig.savefig(OUT/"fig3_ours_confusion.png",dpi=300,bbox_inches="tight"); plt.close(fig)

    # 4. paired effect sizes across repeated CV; bootstrap repeat deltas
    pivot=pd.read_csv(OUT/"cv_repeat_metrics.csv").pivot(index="repeat",columns="model",values="macro_f1")
    baselines=[m for m in MODEL_ORDER if m!=name]
    rng=np.random.default_rng(SEED)
    means=[]; los=[]; his=[]
    for m in baselines:
        d=(pivot[name]-pivot[m]).to_numpy()
        boot=np.array([rng.choice(d,size=len(d),replace=True).mean() for _ in range(10000)])
        means.append(d.mean()); los.append(np.percentile(boot,2.5)); his.append(np.percentile(boot,97.5))
    y=np.arange(len(baselines)); fig,ax=plt.subplots(figsize=(7.2,4.8))
    xerr=np.vstack([np.array(means)-np.array(los),np.array(his)-np.array(means)])
    ax.errorbar(means,y,xerr=xerr,fmt="o",capsize=4)
    ax.axvline(0,linewidth=1)
    ax.set_yticks(y,baselines); ax.set_xlabel("Paired Δ Macro-F1: Ours − baseline")
    ax.set_title("Matched repeated-CV effect sizes (bootstrap 95% CI)")
    ax.grid(True,axis="x",alpha=.25); fig.tight_layout()
    fig.savefig(OUT/"fig4_paired_effects.pdf",bbox_inches="tight")
    fig.savefig(OUT/"fig4_paired_effects.png",dpi=300,bbox_inches="tight"); plt.close(fig)

    # 5. reliability diagram for ours vs strongest non-ours by macro-F1
    best_base=summary[summary.model!=name].sort_values("macro_f1_mean",ascending=False).iloc[0].model
    fig,ax=plt.subplots(figsize=(6.2,5.0)); ax.plot([0,1],[0,1],linestyle="--",label="Ideal")
    reliability_rows=[]
    for m in [name,best_base]:
        qq=preds[(preds.repeat==0)&(preds.model==m)].sort_values("sample_idx")
        yy=qq.y_true.to_numpy(int); pp=qq[[f"p{i}" for i in range(n_classes)]].to_numpy(float)
        ece,rows=ece_toplabel(yy,pp,n_bins=6)
        xs=[];ys=[]
        for b,lo,hi,n,c,a in rows:
            xs.append(c);ys.append(a);reliability_rows.append({"model":m,"bin":b,"n":n,"mean_confidence":c,"accuracy":a})
        ax.plot(xs,ys,marker="o",label=f"{m} (ECE={ece:.3f})")
    pd.DataFrame(reliability_rows).to_csv(OUT/"reliability_bins.csv",index=False)
    ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_xlabel("Mean confidence");ax.set_ylabel("Empirical accuracy")
    ax.set_title("Out-of-fold reliability on real UCI 308 data")
    ax.grid(True,alpha=.25);ax.legend(fontsize=8,frameon=False);fig.tight_layout()
    fig.savefig(OUT/"fig5_reliability.pdf",bbox_inches="tight")
    fig.savefig(OUT/"fig5_reliability.png",dpi=300,bbox_inches="tight");plt.close(fig)
    return best_base


def fmt(x,d=3):
    if pd.isna(x): return "--"
    return f"{x:.{d}f}"


def write_tex(summary,batchdf,stats,glob,best_base,classes):
    s=summary.set_index("model")
    ours=s.loc["Attention-BiLSTM (Ours)"]
    # identify best baseline in repeated CV
    base=summary[summary.model!="Attention-BiLSTM (Ours)"].sort_values("macro_f1_mean",ascending=False).iloc[0]
    delta=ours.macro_f1_mean-base.macro_f1_mean
    # exact stats row against best baseline
    st=stats[stats.baseline==base.model].iloc[0]

    abstract=f'''\\begin{{abstract}}
This study evaluates temporal volatile-organic-compound (VOC) representation learning using real respiration-modulated chemical-sensor data rather than synthetic signals. Experiments were conducted on the public UCI Gas Sensor Array under Flow Modulation dataset, comprising 58 real measurements from 16 metal-oxide sensors acquired under mechanically simulated respiration. The first 13 respiratory cycles were represented as a $13\\times32$ sequence containing low- and high-frequency responses from all sensors. Seven models---SVM, random forest, XGBoost, LSTM, BiLSTM, CNN--LSTM, and the proposed Attention-BiLSTM---were evaluated on identical repeated stratified five-fold splits, with an additional leave-one-batch-out protocol to quantify session/domain shift. The proposed model obtained a mean macro-F1 of {ours.macro_f1_mean:.3f}, balanced accuracy of {ours.balanced_accuracy_mean:.3f}, and macro-AUC of {ours.macro_auc_mean:.3f} across repeated cross-validation. Under natural batch holdout, its mean macro-F1 was {ours.batch_macro_f1_mean:.3f} with a worst-batch value of {ours.batch_macro_f1_worst:.3f}. Relative to the strongest non-attention baseline ({base.model}), the paired macro-F1 difference was {delta:+.3f}. Calibration and paired out-of-fold significance analyses were additionally performed to distinguish discrimination gains from confidence miscalibration. These results establish a real-data engineering validation of respiration-coupled VOC sequence modeling. Because UCI 308 contains gas-mixture rather than Parkinson's disease labels, the results validate the sensing and representation-learning pipeline but do not constitute clinical PD validation.
\\end{{abstract}}'''
    (OUT/"abstract_real_uci308.tex").write_text(abstract,encoding="utf-8")

    table_rows=[]
    for m in MODEL_ORDER:
        r=s.loc[m]
        bold="\\textbf{" if m=="Attention-BiLSTM (Ours)" else ""
        close="}" if bold else ""
        table_rows.append(f"{bold}{m}{close} & {r.macro_f1_mean:.3f}$\\pm${r.macro_f1_sd:.3f} & {r.balanced_accuracy_mean:.3f} & {r.macro_auc_mean:.3f} & {r.ece_mean:.3f} & {r.nll_mean:.3f} & {r.batch_macro_f1_mean:.3f} & {r.batch_macro_f1_worst:.3f} \\\\")
    table='\n'.join(table_rows)

    sec=f'''\\section{{Results and Analysis}}
\\label{{sec:results}}

\\subsection{{Real-Data Benchmark and Leakage-Controlled Protocol}}

All model comparisons in this section were recomputed on the same real public dataset and the same train/test indices. UCI 308 contains 58 measurements acquired from 16 MOX sensors while a mechanical ventilator imposed an approximately five-breath-per-minute flow modulation. For each measurement, the first 13 respiratory cycles provide a low-frequency magnitude and a respiration-synchronous high-frequency amplitude for each sensor. We therefore formed a $13\\times32$ sequence per measurement and predicted the official four-category \\texttt{{gas}} label (air, acetone, ethanol, or mixture). No synthetic samples were introduced.

Two complementary protocols were used. First, repeated stratified five-fold cross-validation (five repeats; 25 matched folds) measured in-distribution discrimination and calibration. Second, leave-one-batch-out evaluation used the five acquisition batches as natural domain-shift folds, preventing measurements from the held-out day/session from appearing in training. All sequence models and classical baselines used exactly the same folds. Standardization statistics were fitted on training data only; the deep models used an internal stratified validation subset only for early stopping.

\\begin{{table*}}[t]
\\centering
\\caption{{Matched real-data benchmark on UCI 308. Repeated-CV values are aggregated at the repeat level; batch columns summarize leave-one-batch-out generalization. ECE and NLL are lower-is-better.}}
\\label{{tab:strong-real}}
\\small
\\begin{{tabular}}{{lccccccc}}
\\toprule
Model & Macro-F1 & Bal. Acc. & Macro-AUC & ECE & NLL & Batch F1 & Worst Batch F1 \\\\
\\midrule
{table}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}

\\subsection{{Discrimination--Calibration Frontier}}

Table~\\ref{{tab:strong-real}} shows that the proposed Attention-BiLSTM achieved mean macro-F1={ours.macro_f1_mean:.3f}, balanced accuracy={ours.balanced_accuracy_mean:.3f}, and macro-AUC={ours.macro_auc_mean:.3f}. The strongest non-attention comparator by macro-F1 was {base.model} ({base.macro_f1_mean:.3f}), producing a paired mean difference of {delta:+.3f}. Figure~\\ref{{fig:cal-frontier}} evaluates whether discrimination was obtained at the expense of probability quality. This is critical for a screening-oriented model because two classifiers with similar macro-F1 can have very different confidence reliability.

\\begin{{figure}}[t]
\\centering
\\includegraphics[width=\\columnwidth]{{fig1_performance_calibration_frontier.pdf}}
\\caption{{Discrimination--calibration frontier across identical repeated-CV splits. The desired region is high macro-F1 and low expected calibration error (ECE).}}
\\label{{fig:cal-frontier}}
\\end{{figure}}

\\subsection{{Natural Batch-Shift Robustness}}

Random stratified CV can overestimate deployability when chemical sensors drift across sessions. UCI 308 contains five acquisition batches collected across several days, allowing a stricter leave-one-batch-out stress test. The proposed model obtained mean held-out-batch macro-F1={ours.batch_macro_f1_mean:.3f}$\\pm${ours.batch_macro_f1_sd:.3f}; its worst batch was {ours.batch_macro_f1_worst:.3f}. The difference between repeated-CV and batch-holdout macro-F1 was {ours.robustness_gap_f1:+.3f}, quantifying the practical domain-shift penalty rather than hiding it behind a random split.

\\begin{{figure}}[t]
\\centering
\\includegraphics[width=\\columnwidth]{{fig2_batch_robustness.pdf}}
\\caption{{Leave-one-batch-out macro-F1 trajectories. Each point is a completely unseen acquisition batch, exposing session-specific sensor drift and model fragility.}}
\\label{{fig:batch-robust}}
\\end{{figure}}

\\subsection{{Error Structure of the Proposed Attention-BiLSTM}}

Figure~\\ref{{fig:ours-cm}} reports row-normalized out-of-fold predictions from the first complete five-fold repetition, so every measurement contributes exactly once and no training prediction appears in the matrix. This view is more informative than global accuracy because it reveals whether errors are concentrated in a particular chemical state, especially mixture-versus-pure-analyte confusions.

\\begin{{figure}}[t]
\\centering
\\includegraphics[width=0.92\\columnwidth]{{fig3_ours_confusion.pdf}}
\\caption{{Row-normalized first-repeat out-of-fold confusion matrix for the proposed Attention-BiLSTM.}}
\\label{{fig:ours-cm}}
\\end{{figure}}

\\subsection{{Paired Effect Size and Statistical Evidence}}

A Friedman test across the seven models using repeat-level macro-F1 gave $\\chi^2={glob['friedman_statistic']:.3f}$ ($p={glob['friedman_p']:.4g}$). For the strongest baseline ({base.model}), the repeat-level paired Wilcoxon comparison gave $p={st.wilcoxon_p:.4g}$ (Holm-adjusted $p={st.wilcoxon_holm_p:.4g}$). On the first-repeat out-of-fold predictions, exact McNemar analysis counted {int(st.mcnemar_ours_correct_baseline_wrong)} cases correct only by the proposed model and {int(st.mcnemar_ours_wrong_baseline_correct)} cases correct only by {base.model}, yielding $p={st.mcnemar_exact_p:.4g}$. A paired label-swap permutation test on macro-F1 yielded an observed difference of {st.oof_delta_macro_f1:+.3f} with $p={st.paired_perm_p_one_sided:.4g}$. Because repeated cross-validation folds are correlated, effect size and out-of-fold paired tests are reported alongside, rather than relying on fold-wise $p$-values alone.

\\begin{{figure}}[t]
\\centering
\\includegraphics[width=\\columnwidth]{{fig4_paired_effects.pdf}}
\\caption{{Matched repeated-CV effect sizes. Points show mean paired $\\Delta$macro-F1 (Attention-BiLSTM minus baseline); intervals are bootstrap 95\\% confidence intervals over repeat-level differences.}}
\\label{{fig:paired-effects}}
\\end{{figure}}

\\subsection{{Calibration Reliability}}

Figure~\\ref{{fig:reliability}} compares the proposed model with the strongest non-attention baseline ({best_base}) using first-repeat out-of-fold probabilities. This analysis separates correct ranking from trustworthy confidence, an important distinction when model scores are intended to drive a downstream screening threshold.

\\begin{{figure}}[t]
\\centering
\\includegraphics[width=\\columnwidth]{{fig5_reliability.pdf}}
\\caption{{Out-of-fold reliability diagram for the proposed model and the strongest non-attention baseline.}}
\\label{{fig:reliability}}
\\end{{figure}}

\\subsection{{Engineering Interpretation and Clinical Boundary}}

The experiment demonstrates that bidirectional and attention-based temporal models can be evaluated rigorously on real respiration-coupled MOX sequences without synthetic data. More importantly, the batch-holdout experiment quantifies how much performance survives an acquisition-session shift. However, UCI 308 contains controlled acetone/ethanol gas classes rather than Parkinson's disease labels. The present results therefore validate the real-data temporal sensing and representation-learning pipeline, not clinical PD diagnosis. A clinically valid early-PD claim still requires a paired cohort of PD/prodromal-risk/control participants acquired with the same synchronized respiratory and VOC sensing protocol.
'''
    (OUT/"results_strong_real_uci308.tex").write_text(sec,encoding="utf-8")

    bib=r'''\\bibitem{uci308}
A. Ziyatdinov and J. Fonollosa, ``Gas sensor array under flow modulation,'' UCI Machine Learning Repository, 2015, doi: 10.24432/C5BG7G.

\\bibitem{ziyatdinov2015dib}
A. Ziyatdinov, J. Fonollosa, L. Fern\\'andez, A. Pomares, R. Huerta, and S. Marco, ``Data set from gas sensor array under flow modulation,'' \\emph{Data in Brief}, vol. 3, pp. 131--136, 2015, doi: 10.1016/j.dib.2015.02.016.
'''
    (OUT/"bibliography_real_uci308.tex").write_text(bib,encoding="utf-8")


def main():
    seed_all(SEED)
    df=download_features()
    df.to_csv(OUT/"uci308_features_used.csv",index=False)
    X,y,groups,le,meta=build_tensor(df)
    meta["y_encoded"]=y
    meta.to_csv(OUT/"sample_metadata.csv",index=False)
    classes=list(le.classes_)
    (OUT/"dataset_manifest.json").write_text(json.dumps({
        "dataset":"UCI 308 Gas sensor array under flow modulation",
        "doi":"10.24432/C5BG7G","n_samples":int(len(X)),"sequence_shape":list(X.shape[1:]),
        "classes":classes,"class_counts":pd.Series(le.inverse_transform(y)).value_counts().to_dict(),
        "batches":pd.Series(groups).value_counts().to_dict(),"synthetic_data":False,
        "target":"gas (4-class)","cv":"5x repeated stratified 5-fold + leave-one-batch-out",
        "seed":SEED
    },indent=2),encoding="utf-8")

    folds,preds,repdf=run_repeated_cv(X,y,len(classes))
    batchdf,batchpred=run_logo(X,y,groups,len(classes))
    st,glob=stats_tests(repdf,preds,len(classes))
    summary=make_summaries(repdf,preds,batchdf,len(classes),classes)
    best_base=figures(summary,preds,batchdf,st,len(classes),classes)
    write_tex(summary,batchdf,st,glob,best_base,classes)
    print("\n=== MODEL SUMMARY ===")
    print(summary.to_string(index=False))
    print("\n=== PAIRED STATS ===")
    print(st.to_string(index=False))
    print("\n=== GLOBAL ===")
    print(json.dumps(glob,indent=2))

if __name__=="__main__":
    main()
