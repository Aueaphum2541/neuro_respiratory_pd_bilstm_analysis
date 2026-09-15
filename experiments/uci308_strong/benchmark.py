#!/usr/bin/env python3
from __future__ import annotations
import os, json, random, time, math, urllib.request, zipfile, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
    roc_auc_score, confusion_matrix, log_loss)
from xgboost import XGBClassifier
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

SEED=260915
REPEATS=20
FOLDS=5
DEVICE=torch.device('cpu')
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
OUT.mkdir(parents=True,exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(max(1,min(4,os.cpu_count() or 2)))

DATA_URL='https://cdn.uci-ics-mlr-prod.aws.uci.edu/308/gas%2Bsensor%2Barray%2Bunder%2Bflow%2Bmodulation.zip'


def download_data():
    z=ROOT/'uci308.zip'; data_dir=ROOT/'data'
    data_dir.mkdir(exist_ok=True)
    f=data_dir/'features.csv'
    if f.exists(): return f
    print('Downloading UCI 308...',flush=True)
    urllib.request.urlretrieve(DATA_URL,z)
    with zipfile.ZipFile(z) as zz: zz.extractall(data_dir)
    hits=list(data_dir.rglob('features.csv'))
    if not hits: raise FileNotFoundError('features.csv not found after extraction')
    if hits[0]!=f: shutil.copy2(hits[0],f)
    return f


def load_tensor(path):
    d=pd.read_csv(path)
    req=['exp','batch','gas','lab']
    for c in req:
        if c not in d.columns: raise KeyError(c)
    blocks=[]
    for cyc in range(1,14):
        cols=[]
        for s in range(1,17):
            cols += [f'S{s}_r{cyc}_Alf',f'S{s}_r{cyc}_Ahf']
        miss=[c for c in cols if c not in d.columns]
        if miss: raise KeyError(miss[:5])
        blocks.append(d[cols].to_numpy(np.float32))
    X=np.stack(blocks,axis=1)  # N x 13 x 32
    le=LabelEncoder(); y=le.fit_transform(d['gas'].astype(str))
    batches=d['batch'].astype(str).to_numpy()
    meta=d[['exp','batch','gas','lab']].copy()
    return X,y,batches,meta,le


def multiclass_brier(y,p,k):
    oh=np.eye(k,dtype=float)[y]
    return float(np.mean(np.sum((p-oh)**2,axis=1)))


def multiclass_ece(y,p,n_bins=10):
    conf=p.max(1); pred=p.argmax(1); corr=(pred==y).astype(float)
    edges=np.linspace(0,1,n_bins+1); e=0.0
    for i in range(n_bins):
        m=(conf>=edges[i]) & (conf < edges[i+1] if i<n_bins-1 else conf<=edges[i+1])
        if m.any(): e += m.mean()*abs(corr[m].mean()-conf[m].mean())
    return float(e)


def metrics(y,p):
    pred=p.argmax(1); k=p.shape[1]
    try: auc=roc_auc_score(y,p,multi_class='ovr',average='macro')
    except Exception: auc=np.nan
    return dict(
        accuracy=accuracy_score(y,pred),
        balanced_accuracy=balanced_accuracy_score(y,pred),
        macro_f1=f1_score(y,pred,average='macro',zero_division=0),
        macro_auc=auc,
        logloss=log_loss(y,p,labels=np.arange(k)),
        brier=multiclass_brier(y,p,k),
        ece=multiclass_ece(y,p),
    )


def make_inner_split(tr,y,seed):
    yy=y[tr]
    try:
        s=StratifiedShuffleSplit(n_splits=1,test_size=.22,random_state=seed)
        a,b=next(s.split(np.zeros(len(tr)),yy))
        return tr[a],tr[b]
    except Exception:
        rng=np.random.default_rng(seed); q=tr.copy(); rng.shuffle(q)
        n=max(4,int(round(.22*len(q))))
        return q[n:],q[:n]


def inv_class_weights(y,k):
    cnt=np.bincount(y,minlength=k).astype(float)
    w=len(y)/(k*np.maximum(cnt,1))
    return w


class LSTMNet(nn.Module):
    def __init__(self,d,k,bidir=False,attention=False):
        super().__init__(); self.attention=attention
        self.rnn=nn.LSTM(d,40,batch_first=True,bidirectional=bidir)
        h=40*(2 if bidir else 1)
        if attention:
            self.attn=nn.Sequential(nn.Linear(h,32),nn.Tanh(),nn.Linear(32,1,bias=False))
        self.head=nn.Sequential(nn.Dropout(.25),nn.Linear(h,32),nn.ReLU(),nn.Dropout(.15),nn.Linear(32,k))
    def forward(self,x,return_attention=False):
        h,_=self.rnn(x)
        if self.attention:
            a=torch.softmax(self.attn(h).squeeze(-1),1); z=(h*a.unsqueeze(-1)).sum(1)
        else:
            a=torch.full((x.shape[0],x.shape[1]),1.0/x.shape[1],device=x.device); z=h.mean(1)
        out=self.head(z)
        return (out,a) if return_attention else out


class CNNLSTMNet(nn.Module):
    def __init__(self,d,k):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv1d(d,64,3,padding=1),nn.BatchNorm1d(64),nn.ReLU(),nn.Dropout(.15),
                                nn.Conv1d(64,64,3,padding=1),nn.ReLU())
        self.rnn=nn.LSTM(64,48,batch_first=True)
        self.head=nn.Sequential(nn.Dropout(.25),nn.Linear(48,32),nn.ReLU(),nn.Linear(32,k))
    def forward(self,x,return_attention=False):
        z=self.conv(x.transpose(1,2)).transpose(1,2); h,_=self.rnn(z); pooled=h.mean(1); out=self.head(pooled)
        a=torch.full((x.shape[0],x.shape[1]),1.0/x.shape[1],device=x.device)
        return (out,a) if return_attention else out


def scale_seq(X,fit_idx,*idxs):
    mu=X[fit_idx].mean((0,1),keepdims=True); sd=X[fit_idx].std((0,1),keepdims=True)+1e-6
    return [((X[ii]-mu)/sd).astype(np.float32) for ii in idxs]


def train_deep(X,y,fit_idx,val_idx,test_idx,kind,seed,k):
    xfit,xval,xte=scale_seq(X,fit_idx,fit_idx,val_idx,test_idx)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if kind=='LSTM': model=LSTMNet(X.shape[-1],k,False,False)
    elif kind=='BiLSTM': model=LSTMNet(X.shape[-1],k,True,False)
    elif kind=='Attention-BiLSTM': model=LSTMNet(X.shape[-1],k,True,True)
    elif kind=='CNN-LSTM': model=CNNLSTMNet(X.shape[-1],k)
    else: raise ValueError(kind)
    model.to(DEVICE)
    cw=torch.tensor(inv_class_weights(y[fit_idx],k),dtype=torch.float32,device=DEVICE)
    crit=nn.CrossEntropyLoss(weight=cw)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    dl=DataLoader(TensorDataset(torch.tensor(xfit),torch.tensor(y[fit_idx],dtype=torch.long)),batch_size=16,shuffle=True)
    xv=torch.tensor(xval,device=DEVICE); yv=torch.tensor(y[val_idx],dtype=torch.long,device=DEVICE)
    best=None; best_loss=float('inf'); stale=0; epochs=0
    for ep in range(120):
        epochs=ep+1; model.train()
        for xb,yb in dl:
            xb=xb.to(DEVICE); yb=yb.to(DEVICE); opt.zero_grad(); loss=crit(model(xb),yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        model.eval()
        with torch.no_grad(): vl=float(crit(model(xv),yv).item())
        if vl < best_loss-1e-4:
            best_loss=vl; best={n:t.detach().cpu().clone() for n,t in model.state_dict().items()}; stale=0
        else: stale+=1
        if stale>=15: break
    model.load_state_dict(best); model.eval()
    with torch.no_grad():
        logits,a=model(torch.tensor(xte,device=DEVICE),return_attention=True)
        p=torch.softmax(logits,1).cpu().numpy(); a=a.cpu().numpy()
    return p,a,epochs,sum(q.numel() for q in model.parameters())


def classical_models(seed,k):
    return {
      'SVM': Pipeline([('sc',StandardScaler()),('m',SVC(C=10,kernel='rbf',gamma='scale',probability=True,class_weight='balanced',random_state=seed))]),
      'RF': RandomForestClassifier(n_estimators=600,max_features='sqrt',min_samples_leaf=1,class_weight='balanced_subsample',random_state=seed,n_jobs=-1),
      'XGBoost': XGBClassifier(n_estimators=350,max_depth=3,learning_rate=.03,subsample=.9,colsample_bytree=.9,
                    reg_lambda=1.5,reg_alpha=.05,objective='multi:softprob',num_class=k,eval_metric='mlogloss',random_state=seed,n_jobs=4),
    }


def fit_classical(name,m,Xtr,ytr,Xte,k):
    if name=='XGBoost':
        w=inv_class_weights(ytr,k)[ytr]; m.fit(Xtr,ytr,sample_weight=w)
    else: m.fit(Xtr,ytr)
    p=m.predict_proba(Xte)
    if p.shape[1]!=k:
        full=np.zeros((len(Xte),k)); full[:,m.classes_.astype(int)]=p; p=full
    return p


def ci95(v):
    v=np.asarray(v,float); n=len(v); mean=v.mean(); se=v.std(ddof=1)/math.sqrt(n)
    d=stats.t.ppf(.975,n-1)*se
    return mean,mean-d,mean+d


def holm(ps):
    ps=np.asarray(ps,float); n=len(ps); order=np.argsort(ps); out=np.empty(n); prev=0
    for rank,idx in enumerate(order):
        val=min(1.0,(n-rank)*ps[idx]); val=max(val,prev); out[idx]=val; prev=val
    return out


def run_main_cv(X,y,meta,le):
    k=len(le.classes_); Xtab=X.reshape(len(X),-1)
    splitter=RepeatedStratifiedKFold(n_splits=FOLDS,n_repeats=REPEATS,random_state=SEED)
    fold_rows=[]; pred_rows=[]; att_rows=[]; diag=[]
    names=['SVM','RF','XGBoost','LSTM','BiLSTM','Attention-BiLSTM','CNN-LSTM']
    for split_id,(tr,te) in enumerate(splitter.split(Xtab,y)):
        rep=split_id//FOLDS; fold=split_id%FOLDS
        fit_idx,val_idx=make_inner_split(tr,y,SEED+split_id)
        print(f'Main CV {split_id+1}/{REPEATS*FOLDS}',flush=True)
        for name,m in classical_models(SEED+split_id,k).items():
            t0=time.perf_counter(); p=fit_classical(name,m,Xtab[tr],y[tr],Xtab[te],k); sec=time.perf_counter()-t0
            mm=metrics(y[te],p); fold_rows.append(dict(repeat=rep,fold=fold,split_id=split_id,model=name,**mm))
            diag.append(dict(split_id=split_id,model=name,train_seconds=sec,epochs=np.nan,parameters=np.nan))
            for j,ii in enumerate(te): pred_rows.append(dict(repeat=rep,fold=fold,split_id=split_id,model=name,sample=int(ii),y_true=int(y[ii]),**{f'p{c}':float(p[j,c]) for c in range(k)}))
        for di,name in enumerate(['LSTM','BiLSTM','Attention-BiLSTM','CNN-LSTM']):
            t0=time.perf_counter(); p,a,ep,par=train_deep(X,y,fit_idx,val_idx,te,name,SEED+split_id*17+di,k); sec=time.perf_counter()-t0
            mm=metrics(y[te],p); fold_rows.append(dict(repeat=rep,fold=fold,split_id=split_id,model=name,**mm))
            diag.append(dict(split_id=split_id,model=name,train_seconds=sec,epochs=ep,parameters=par))
            for j,ii in enumerate(te):
                pred_rows.append(dict(repeat=rep,fold=fold,split_id=split_id,model=name,sample=int(ii),y_true=int(y[ii]),**{f'p{c}':float(p[j,c]) for c in range(k)}))
                if name=='Attention-BiLSTM': att_rows.append(dict(repeat=rep,fold=fold,sample=int(ii),y_true=int(y[ii]),attention=json.dumps(a[j].tolist())))
    folds=pd.DataFrame(fold_rows); preds=pd.DataFrame(pred_rows); atts=pd.DataFrame(att_rows); diagnostics=pd.DataFrame(diag)
    folds.to_csv(OUT/'fold_metrics.csv',index=False); preds.to_csv(OUT/'oof_predictions_repeated.csv',index=False); atts.to_csv(OUT/'attention_oof.csv',index=False); diagnostics.to_csv(OUT/'training_diagnostics.csv',index=False)
    rows=[]
    for name,g in folds.groupby('model'):
        r={'model':name}
        for c in ['accuracy','balanced_accuracy','macro_f1','macro_auc','logloss','brier','ece']:
            vals=g[c].dropna().to_numpy(); mean,lo,hi=ci95(vals); r[c]=mean; r[c+'_sd']=vals.std(ddof=1); r[c+'_lo']=lo; r[c+'_hi']=hi
        rows.append(r)
    summary=pd.DataFrame(rows).sort_values('macro_f1',ascending=False); summary.to_csv(OUT/'model_summary.csv',index=False)
    # paired tests vs ours using exactly the same 100 outer test folds
    ours=folds[folds.model=='Attention-BiLSTM'].sort_values('split_id')
    tests=[]
    for name in names:
        if name=='Attention-BiLSTM': continue
        g=folds[folds.model==name].sort_values('split_id')
        diff=ours.macro_f1.to_numpy()-g.macro_f1.to_numpy(); md,lo,hi=ci95(diff)
        try: wp=stats.wilcoxon(diff,zero_method='zsplit',alternative='two-sided').pvalue
        except Exception: wp=1.0
        tp=stats.ttest_rel(ours.accuracy,g.accuracy).pvalue
        dz=float(diff.mean()/(diff.std(ddof=1)+1e-12))
        tests.append(dict(baseline=name,delta_macro_f1=md,ci_lo=lo,ci_hi=hi,wilcoxon_p=wp,paired_accuracy_t_p=tp,cohen_dz=dz))
    tdf=pd.DataFrame(tests); tdf['holm_p']=holm(tdf.wilcoxon_p.to_numpy()); tdf.to_csv(OUT/'paired_tests.csv',index=False)
    return folds,preds,atts,summary,tdf


def average_sample_predictions(preds,k):
    pcols=[f'p{i}' for i in range(k)]
    agg=preds.groupby(['model','sample','y_true'],as_index=False)[pcols].mean()
    return agg


def run_batch_robustness(X,y,batches,le):
    k=len(le.classes_); Xtab=X.reshape(len(X),-1); rows=[]
    for bi,b in enumerate(pd.unique(batches)):
        te=np.where(batches==b)[0]; tr=np.where(batches!=b)[0]; fit_idx,val_idx=make_inner_split(tr,y,SEED+900+bi)
        for name,m in classical_models(SEED+900+bi,k).items():
            p=fit_classical(name,m,Xtab[tr],y[tr],Xtab[te],k); mm=metrics(y[te],p); rows.append(dict(batch=b,model=name,n_test=len(te),**mm))
        for di,name in enumerate(['LSTM','BiLSTM','Attention-BiLSTM','CNN-LSTM']):
            p,_,_,_=train_deep(X,y,fit_idx,val_idx,te,name,SEED+1200+bi*13+di,k); mm=metrics(y[te],p); rows.append(dict(batch=b,model=name,n_test=len(te),**mm))
    d=pd.DataFrame(rows); d.to_csv(OUT/'leave_one_batch_out.csv',index=False); return d


def make_figures(folds,preds,atts,summary,tests,batch,X,y,meta,le):
    # 1. Macro-F1 forest plot
    s=summary.sort_values('macro_f1')
    fig,ax=plt.subplots(figsize=(7.2,4.8)); yy=np.arange(len(s)); v=s.macro_f1.to_numpy(); lo=s.macro_f1_lo.to_numpy(); hi=s.macro_f1_hi.to_numpy()
    ax.errorbar(v,yy,xerr=np.vstack([v-lo,hi-v]),fmt='o',capsize=3,lw=1.4)
    ax.set_yticks(yy,s.model); ax.set_xlabel('Macro-F1 (mean and 95% CI across 100 shared folds)'); ax.set_title('UCI 308: shared-split discrimination with uncertainty'); ax.grid(alpha=.22)
    fig.tight_layout(); fig.savefig(OUT/'fig1_macro_f1_forest.pdf',bbox_inches='tight'); fig.savefig(OUT/'fig1_macro_f1_forest.png',dpi=300,bbox_inches='tight'); plt.close(fig)
    # 2. Pairwise delta forest
    t=tests.sort_values('delta_macro_f1'); fig,ax=plt.subplots(figsize=(7.2,4.3)); yy=np.arange(len(t)); v=t.delta_macro_f1.to_numpy(); lo=t.ci_lo.to_numpy(); hi=t.ci_hi.to_numpy()
    ax.axvline(0,ls='--',lw=1); ax.errorbar(v,yy,xerr=np.vstack([v-lo,hi-v]),fmt='o',capsize=3,lw=1.4)
    labels=[f"{n}  (Holm p={p:.3g})" for n,p in zip(t.baseline,t.holm_p)]
    ax.set_yticks(yy,labels); ax.set_xlabel('Δ Macro-F1: Attention-BiLSTM − baseline'); ax.set_title('Paired effect sizes under identical test folds'); ax.grid(alpha=.22)
    fig.tight_layout(); fig.savefig(OUT/'fig2_paired_delta_f1.pdf',bbox_inches='tight'); fig.savefig(OUT/'fig2_paired_delta_f1.png',dpi=300,bbox_inches='tight'); plt.close(fig)
    # 3. Batch robustness heat map
    piv=batch.pivot(index='model',columns='batch',values='macro_f1').reindex(summary.model.tolist())
    fig,ax=plt.subplots(figsize=(8.0,4.8)); im=ax.imshow(piv.to_numpy(),aspect='auto',vmin=0,vmax=1)
    ax.set_yticks(np.arange(len(piv)),piv.index); ax.set_xticks(np.arange(len(piv.columns)),piv.columns,rotation=30,ha='right'); ax.set_title('Leave-one-batch-out Macro-F1: temporal/domain-shift robustness')
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]): ax.text(j,i,f'{piv.iloc[i,j]:.2f}',ha='center',va='center',fontsize=8)
    fig.colorbar(im,ax=ax,label='Macro-F1'); fig.tight_layout(); fig.savefig(OUT/'fig3_batch_robustness.pdf',bbox_inches='tight'); fig.savefig(OUT/'fig3_batch_robustness.png',dpi=300,bbox_inches='tight'); plt.close(fig)
    # 4. Attention profile, averaged per sample then per class
    att=atts.copy(); att['vec']=att.attention.map(json.loads); per=att.groupby(['sample','y_true']).vec.apply(lambda z:np.mean(np.stack(z),axis=0)).reset_index()
    fig,ax=plt.subplots(figsize=(7.4,4.8)); cycles=np.arange(1,14)
    for c,label in enumerate(le.classes_):
        arr=np.stack(per[per.y_true==c].vec.to_list()); m=arr.mean(0); sd=arr.std(0)
        ax.plot(cycles,m,marker='o',lw=1.5,label=str(label)); ax.fill_between(cycles,np.maximum(0,m-sd),m+sd,alpha=.12)
    ax.set_xticks(cycles); ax.set_xlabel('Respiratory cycle'); ax.set_ylabel('Attention weight'); ax.set_title('Attention-BiLSTM: class-conditioned temporal saliency'); ax.grid(alpha=.22); ax.legend(frameon=False,fontsize=8)
    fig.tight_layout(); fig.savefig(OUT/'fig4_attention_profile.pdf',bbox_inches='tight'); fig.savefig(OUT/'fig4_attention_profile.png',dpi=300,bbox_inches='tight'); plt.close(fig)
    # 5. OOF averaged confusion matrix for ours
    agg=average_sample_predictions(preds,len(le.classes_)); d=agg[agg.model=='Attention-BiLSTM'].sort_values('sample'); p=d[[f'p{i}' for i in range(len(le.classes_))]].to_numpy(); pred=p.argmax(1); cm=confusion_matrix(d.y_true,pred,labels=np.arange(len(le.classes_)),normalize='true')
    fig,ax=plt.subplots(figsize=(5.7,5.1)); im=ax.imshow(cm,vmin=0,vmax=1); ax.set_xticks(np.arange(len(le.classes_)),le.classes_,rotation=35,ha='right'); ax.set_yticks(np.arange(len(le.classes_)),le.classes_); ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title('Attention-BiLSTM repeated-CV consensus confusion matrix')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]): ax.text(j,i,f'{cm[i,j]:.2f}',ha='center',va='center',fontsize=9)
    fig.colorbar(im,ax=ax,label='Recall-normalized fraction'); fig.tight_layout(); fig.savefig(OUT/'fig5_ours_confusion.pdf',bbox_inches='tight'); fig.savefig(OUT/'fig5_ours_confusion.png',dpi=300,bbox_inches='tight'); plt.close(fig)


def make_tex(summary,tests,batch,le):
    s=summary.set_index('model'); ours=s.loc['Attention-BiLSTM']; best=summary.iloc[0]
    best_classical=summary[summary.model.isin(['SVM','RF','XGBoost'])].iloc[0]
    def row(name):
        r=s.loc[name]; return f"{name} & {r.macro_f1:.3f} $\\pm$ {r.macro_f1_sd:.3f} & {r.balanced_accuracy:.3f} & {r.accuracy:.3f} & {r.macro_auc:.3f} & {r.ece:.3f} \\\\"
    table='\n'.join(row(n) for n in ['SVM','RF','XGBoost','LSTM','BiLSTM','CNN-LSTM','Attention-BiLSTM'])
    tb=tests.set_index('baseline')
    def sig(name):
        r=tb.loc[name]; return f"{name} & {r.delta_macro_f1:+.3f} & [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] & {r.holm_p:.3g} & {r.cohen_dz:+.2f} \\\\"
    stat='\n'.join(sig(n) for n in ['LSTM','BiLSTM','CNN-LSTM','SVM','RF','XGBoost'])
    batchmean=batch.groupby('model').macro_f1.mean().sort_values(ascending=False)
    ours_batch=float(batchmean['Attention-BiLSTM']); best_batch_name=str(batchmean.index[0]); best_batch=float(batchmean.iloc[0])
    if best.model=='Attention-BiLSTM': ranktext=f"The proposed Attention-BiLSTM achieved the highest mean Macro-F1 ({ours.macro_f1:.3f}) under the shared repeated-CV protocol."
    else: ranktext=f"The proposed Attention-BiLSTM reached Macro-F1 {ours.macro_f1:.3f}; the strongest model was {best.model} at {best.macro_f1:.3f}. This negative result prevents an unsupported superiority claim and identifies the remaining performance gap."
    tex=rf'''\section{{Results and Analysis}}
\label{{sec:uci308-results}}

\subsection{{Real Public Dataset and Shared Evaluation Protocol}}
All experiments in this section use the real UCI 308 \emph{{Gas Sensor Array under Flow Modulation}} dataset rather than synthetic signals. The dataset contains 58 five-minute measurements from 16 metal-oxide sensors acquired under mechanically modulated respiration. For each measurement, the published feature table provides low-frequency and high-frequency responses for the first 13 respiratory cycles. We therefore reshaped each sample into a $13\times32$ tensor (13 cycles $\times$ 16 sensors $\times$ two frequency components) and used the four-category \texttt{{gas}} label (air, pure acetone, pure ethanol, or binary mixture) as the real-data classification target.

To avoid favorable-split selection, every model was evaluated on exactly the same $20\times5$ repeated stratified cross-validation manifest (100 paired test folds). The compared models were SVM, random forest (RF), XGBoost, LSTM, BiLSTM, CNN--LSTM, and the proposed Attention-BiLSTM. Scaling parameters, early-stopping validation data, and model fitting were derived exclusively from each training fold. Macro-F1 was treated as the primary metric; balanced accuracy, accuracy, macro one-vs-rest ROC-AUC, and expected calibration error (ECE) were also measured.

\subsection{{Shared-Split Model Comparison}}
{ranktext} The best classical baseline was {best_classical.name} with Macro-F1 {best_classical.macro_f1:.3f}. In contrast to a single train/test split, Fig.~\ref{{fig:f1forest}} reports the complete fold-level uncertainty, making variance visible alongside the mean.

\begin{{table*}}[t]
\centering
\caption{{UCI 308 performance under identical $20\times5$ repeated stratified CV. Macro-F1 is shown as mean $\pm$ fold standard deviation.}}
\label{{tab:uci308-main}}
\begin{{tabular}}{{lccccc}}
\toprule
Model & Macro-F1 & Balanced Acc. & Accuracy & Macro AUC & ECE \\
\midrule
{table}
\bottomrule
\end{{tabular}}
\end{{table*}}

\begin{{figure}}[t]
\centering
\includegraphics[width=\columnwidth]{{fig1_macro_f1_forest.pdf}}
\caption{{Macro-F1 mean and 95\% confidence interval across 100 shared outer folds. The plot emphasizes both discrimination and resampling uncertainty.}}
\label{{fig:f1forest}}
\end{{figure}}

\subsection{{Paired Statistical Analysis}}
Because all models share the same 100 test folds, fold-wise differences can be analyzed directly. Table~\ref{{tab:paired}} reports the mean paired Macro-F1 gain of Attention-BiLSTM over each comparator, its 95\% confidence interval, Wilcoxon signed-rank $p$-value after Holm correction, and paired Cohen's $d_z$. Figure~\ref{{fig:paired}} visualizes the same effects. This analysis distinguishes a reproducible improvement from an apparent gain caused by split variance.

\begin{{table*}}[t]
\centering
\caption{{Paired fold-wise comparison: Attention-BiLSTM minus baseline.}}
\label{{tab:paired}}
\begin{{tabular}}{{lcccc}}
\toprule
Baseline & $\Delta$ Macro-F1 & 95\% CI & Holm $p$ & $d_z$ \\
\midrule
{stat}
\bottomrule
\end{{tabular}}
\end{{table*}}

\begin{{figure}}[t]
\centering
\includegraphics[width=\columnwidth]{{fig2_paired_delta_f1.pdf}}
\caption{{Paired Macro-F1 effect sizes under identical test folds. Positive values favor Attention-BiLSTM; confidence intervals crossing zero indicate uncertainty in the direction of the effect.}}
\label{{fig:paired}}
\end{{figure}}

\subsection{{Batch-Shift Robustness}}
Random stratification can overestimate deployment performance when acquisition-session effects leak across folds. UCI 308 provides five acquisition batches collected across multiple sessions. We therefore performed a second, harder leave-one-batch-out experiment in which an entire batch was unseen during training. Mean leave-one-batch-out Macro-F1 for Attention-BiLSTM was {ours_batch:.3f}; the strongest batch-robust model was {best_batch_name} at {best_batch:.3f}. Figure~\ref{{fig:batch}} exposes which models fail under specific session shifts rather than hiding this behavior in a global average.

\begin{{figure}}[t]
\centering
\includegraphics[width=\columnwidth]{{fig3_batch_robustness.pdf}}
\caption{{Leave-one-batch-out Macro-F1. Each column is a completely held-out acquisition batch, providing a direct stress test for temporal/session domain shift.}}
\label{{fig:batch}}
\end{{figure}}

\subsection{{Temporal Interpretability}}
The attention mechanism produces one normalized weight for each of the 13 respiratory cycles. Figure~\ref{{fig:att}} aggregates held-out attention vectors first within sample across repeats and then by gas category. Non-uniform profiles indicate that the recurrent classifier does not weight every cycle equally; class-dependent shifts identify portions of the exposure trajectory that are most informative to the decision. This interpretation is based exclusively on held-out predictions rather than training-set attention.

\begin{{figure}}[t]
\centering
\includegraphics[width=\columnwidth]{{fig4_attention_profile.pdf}}
\caption{{Class-conditioned Attention-BiLSTM temporal saliency across 13 respiratory cycles. Curves are computed only from out-of-fold predictions; shaded regions show one standard deviation across samples.}}
\label{{fig:att}}
\end{{figure}}

\subsection{{Error Structure}}
The repeated-CV consensus confusion matrix in Fig.~\ref{{fig:cm}} averages each sample's probability vector across the 20 repeats before assigning a final class. This avoids visually counting the same sample 20 times and reveals class-specific recall directly. Together with Macro-F1 and balanced accuracy, the matrix provides a stricter view than overall accuracy alone.

\begin{{figure}}[t]
\centering
\includegraphics[width=0.92\columnwidth]{{fig5_ours_confusion.pdf}}
\caption{{Recall-normalized consensus confusion matrix for Attention-BiLSTM after averaging each sample's out-of-fold probabilities across repeated CV.}}
\label{{fig:cm}}
\end{{figure}}

\subsection{{Interpretation Boundary}}
These experiments remove synthetic performance from the validation section and demonstrate the behavior of the proposed temporal architecture on real respiration-modulated chemical-sensor measurements. They validate the engineering hypothesis that cycle-resolved VOC dynamics can be learned by recurrent models. However, UCI 308 contains acetone/ethanol gas-mixture labels rather than Parkinson's disease labels. Therefore, these results are real-data validation of the sensing and temporal-learning pipeline, not clinical PD sensitivity or specificity. A clinical PD claim requires a paired cohort measured with the same respiratory/VOC acquisition protocol.
'''
    (OUT/'results_section.tex').write_text(tex)


def main():
    path=download_data(); X,y,batches,meta,le=load_tensor(path)
    meta.to_csv(OUT/'dataset_manifest.csv',index=False)
    (OUT/'classes.json').write_text(json.dumps({'classes':le.classes_.tolist(),'shape':list(X.shape),'n':len(X),'repeats':REPEATS,'folds':FOLDS},indent=2))
    print('Tensor',X.shape,'classes',dict(zip(le.classes_,np.bincount(y))),'batches',pd.Series(batches).value_counts().to_dict(),flush=True)
    folds,preds,atts,summary,tests=run_main_cv(X,y,meta,le)
    batch=run_batch_robustness(X,y,batches,le)
    make_figures(folds,preds,atts,summary,tests,batch,X,y,meta,le)
    make_tex(summary,tests,batch,le)
    print('\nSUMMARY\n',summary.to_string(index=False),flush=True)
    print('\nPAIRED\n',tests.to_string(index=False),flush=True)
    print('\nBATCH\n',batch.groupby('model')[['macro_f1','balanced_accuracy','accuracy']].mean().sort_values('macro_f1',ascending=False).to_string(),flush=True)

if __name__=='__main__': main()
