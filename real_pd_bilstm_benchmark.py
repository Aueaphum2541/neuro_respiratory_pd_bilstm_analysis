#!/usr/bin/env python3
from __future__ import annotations
import json, os, random, time, warnings
from pathlib import Path
import numpy as np, pandas as pd, requests
from scipy.stats import skew, kurtosis
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss, confusion_matrix, f1_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
warnings.filterwarnings('ignore')
SEED=260915; FS=6.0; N_SAMPLES=10800; WINDOW_N=1800; FRAME_N=30; OUT=Path('real_pd_results'); OUT.mkdir(exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(max(1,min(4,os.cpu_count() or 2)))
REPO='michalandelman/PD-classification-data'; API=f'https://api.github.com/repos/{REPO}/contents'; RAW=f'https://raw.githubusercontent.com/{REPO}/main'

def subjects(group):
    r=requests.get(f'{API}/{group}?ref=main',timeout=30); r.raise_for_status()
    return sorted(x['name'] for x in r.json() if x['name'].endswith('.csv'))

def read_flow(group,name):
    d=pd.read_csv(f'{RAW}/{group}/{name}',usecols=['time','pressure1','pressure2'],nrows=N_SAMPLES,dtype='float32').replace([np.inf,-np.inf],np.nan).dropna()
    if len(d)<N_SAMPLES: raise RuntimeError(f'{name}: {len(d)} valid rows')
    x=((d.pressure2.to_numpy()-d.pressure1.to_numpy())/2).astype('float32')[:N_SAMPLES]
    lo,hi=np.quantile(x,[.005,.995]); x=np.clip(x,lo,hi); return ((x-x.mean())/(x.std()+1e-6)).astype('float32')

def spec(x):
    x=x-x.mean(); p=np.abs(np.fft.rfft(x))**2; f=np.fft.rfftfreq(len(x),1/FS); s=p.sum()+1e-12
    return float((f*p).sum()/s),float(p[(f>=.1)&(f<=.8)].sum()/s)

def feat(x):
    dx=np.diff(x); z=np.mean(np.signbit(x[1:])!=np.signbit(x[:-1])); c,b=spec(x)
    a=[x.mean(),x.std(),np.sqrt(np.mean(x*x)),np.ptp(x),np.mean(np.abs(dx)),z,skew(x,bias=False),kurtosis(x,fisher=True,bias=False),c,b]
    return np.nan_to_num(np.asarray(a,'float32'))

def sequence(w): return np.stack([feat(w[i:i+FRAME_N]) for i in range(0,WINDOW_N,FRAME_N)])
def tabular(w,s):
    a=np.r_[s.mean(0),s.std(0)]; x=w-w.mean(); p=np.abs(np.fft.rfft(x))**2; f=np.fft.rfftfreq(len(x),1/FS); t=p.sum()+1e-12
    bands=[p[(f>=u)&(f<v)].sum()/t for u,v in [(.05,.15),(.15,.35),(.35,.7),(.7,1.2)]]; dom=float(f[np.argmax(p[1:])+1])
    return np.asarray(list(a)+bands+[dom],'float32')

def build():
    xs,xt,y,g,raw,man=[] ,[],[],[],[],[]
    for group,label in [('Control',0),('PD',1)]:
        names=subjects(group); print(group,len(names),flush=True)
        for k,name in enumerate(names,1):
            try: x=read_flow(group,name)
            except Exception as e: print('SKIP',name,e,flush=True); continue
            sid=name.rsplit('.',1)[0]
            for wi in range(6):
                w=x[wi*WINDOW_N:(wi+1)*WINDOW_N]; s=sequence(w); xs.append(s); xt.append(tabular(w,s)); y.append(label); g.append(sid); raw.append(w)
            man.append({'subject':sid,'label':label,'group':group,'windows':6}); print('loaded',group,k,'/',len(names),name,flush=True)
    pd.DataFrame(man).to_csv(OUT/'subject_manifest.csv',index=False)
    return np.asarray(xs,'float32'),np.asarray(xt,'float32'),np.asarray(y),np.asarray(g),np.asarray(raw,'float32')

class RNN(nn.Module):
    def __init__(self,d,h=32,bi=False,att=False):
        super().__init__(); self.att=att; self.r=nn.LSTM(d,h,batch_first=True,bidirectional=bi); z=h*(2 if bi else 1)
        if att:self.a=nn.Sequential(nn.Linear(z,32),nn.Tanh(),nn.Linear(32,1,bias=False))
        self.head=nn.Sequential(nn.Dropout(.3),nn.Linear(z,32),nn.ReLU(),nn.Dropout(.2),nn.Linear(32,1))
    def forward(self,x,ret=False):
        h,_=self.r(x)
        if self.att: a=torch.softmax(self.a(h).squeeze(-1),1); z=(h*a.unsqueeze(-1)).sum(1)
        else: a=torch.full((x.shape[0],x.shape[1]),1/x.shape[1],device=x.device); z=h.mean(1)
        q=self.head(z).squeeze(-1); return (q,a) if ret else q

def deep(X,y,g,tr,te,kind,seed):
    ug=np.unique(g[tr]); sy=np.array([y[np.where(g==u)[0][0]] for u in ug]); a,b=next(StratifiedShuffleSplit(1,test_size=.2,random_state=seed).split(ug,sy)); ga=set(ug[a]); gb=set(ug[b])
    fi=np.array([i for i in tr if g[i] in ga]); va=np.array([i for i in tr if g[i] in gb]); mu=X[fi].mean((0,1),keepdims=True); sd=X[fi].std((0,1),keepdims=True)+1e-6
    xf=(X[fi]-mu)/sd; xv=(X[va]-mu)/sd; xe=(X[te]-mu)/sd; yf=y[fi]; yv=y[va]
    torch.manual_seed(seed); model=RNN(X.shape[-1],32,kind!='LSTM',kind=='Attn-BiLSTM'); opt=torch.optim.AdamW(model.parameters(),1e-3,weight_decay=1e-4)
    pos=max(1,(yf==1).sum()); neg=max(1,(yf==0).sum()); loss=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg/pos],dtype=torch.float32))
    dl=DataLoader(TensorDataset(torch.tensor(xf),torch.tensor(yf,dtype=torch.float32)),32,shuffle=True); xv=torch.tensor(xv); yv=torch.tensor(yv,dtype=torch.float32)
    best=None; bl=1e9; stale=0
    for ep in range(80):
        model.train()
        for xb,yb in dl:
            opt.zero_grad(); l=loss(model(xb),yb); l.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
        model.eval()
        with torch.no_grad(): vl=float(loss(model(xv),yv))
        if vl<bl-1e-4: bl=vl; best={k:v.detach().clone() for k,v in model.state_dict().items()}; stale=0
        else: stale+=1
        if stale>=12:break
    model.load_state_dict(best); model.eval()
    with torch.no_grad(): q,a=model(torch.tensor(xe),True); p=torch.sigmoid(q).numpy(); a=a.numpy()
    return p,a,sum(z.numel() for z in model.parameters()),ep+1

def agg(p,idx,g,y):
    return [(u,int(y[idx[g[idx]==u][0]]),float(np.mean(p[g[idx]==u]))) for u in np.unique(g[idx])]
def ece(y,p,n=6):
    z=0.; b=np.linspace(0,1,n+1)
    for i in range(n):
        m=(p>=b[i])&(p<(b[i+1]) if i<n-1 else p<=b[i+1]); z+=m.mean()*abs(y[m].mean()-p[m].mean()) if m.any() else 0
    return float(z)
def metrics(y,p):
    pr=p>=.5; tn,fp,fn,tp=confusion_matrix(y,pr,labels=[0,1]).ravel()
    return dict(auc=roc_auc_score(y,p),ap=average_precision_score(y,p),accuracy=accuracy_score(y,pr),balanced_accuracy=balanced_accuracy_score(y,pr),sensitivity=tp/(tp+fn),specificity=tn/(tn+fp),f1=f1_score(y,pr),brier=brier_score_loss(y,p),ece=ece(y,p),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))
def boot(y,p,key,B=3000,seed=SEED):
    r=np.random.default_rng(seed); v=[]; n=len(y)
    for _ in range(B):
        i=r.integers(0,n,n); yy=y[i]; pp=p[i]
        if len(np.unique(yy))<2:continue
        if key=='auc': q=roc_auc_score(yy,pp)
        elif key=='ap':q=average_precision_score(yy,pp)
        elif key=='accuracy':q=accuracy_score(yy,pp>=.5)
        else:q=f1_score(yy,pp>=.5,zero_division=0)
        v.append(q)
    return np.percentile(v,[2.5,97.5])
def diff(y,a,b,B=5000):
    r=np.random.default_rng(SEED+9); v=[]; n=len(y)
    for _ in range(B):
        i=r.integers(0,n,n); yy=y[i]
        if len(np.unique(yy))>1:v.append(roc_auc_score(yy,a[i])-roc_auc_score(yy,b[i]))
    v=np.asarray(v); lo,hi=np.percentile(v,[2.5,97.5]); return v.mean(),lo,hi,min(1.,2*min(np.mean(v<=0),np.mean(v>=0)))

def main():
    X,T,y,g,raw=build(); us=np.unique(g); sy=np.array([y[np.where(g==u)[0][0]] for u in us]); print('DATA',X.shape,'subjects',len(us),'PD',sy.sum(),'CTL',(sy==0).sum(),flush=True)
    base={'LR':Pipeline([('s',StandardScaler()),('m',LogisticRegression(max_iter=5000,class_weight='balanced',random_state=SEED))]),'SVM':Pipeline([('s',StandardScaler()),('m',SVC(C=1,kernel='rbf',probability=True,class_weight='balanced',random_state=SEED))]),'RF':RandomForestClassifier(n_estimators=500,min_samples_leaf=2,class_weight='balanced',random_state=SEED,n_jobs=-1),'XGBoost':XGBClassifier(n_estimators=300,max_depth=3,learning_rate=.03,subsample=.85,colsample_bytree=.85,reg_lambda=1,objective='binary:logistic',eval_metric='logloss',random_state=SEED,n_jobs=4)}
    names=list(base)+['LSTM','BiLSTM','Attn-BiLSTM']; oof={n:{} for n in names}; att=[]; diag=[]; par=[]
    cv=StratifiedGroupKFold(5,shuffle=True,random_state=SEED)
    for fold,(tr,te) in enumerate(cv.split(X,y,g),1):
        print('FOLD',fold,'subjects',len(np.unique(g[tr])),len(np.unique(g[te])),flush=True)
        for n,m0 in base.items():
            m=clone(m0); t=time.perf_counter(); m.fit(T[tr],y[tr]); p=m.predict_proba(T[te])[:,1]; diag.append({'fold':fold,'model':n,'fit_seconds':time.perf_counter()-t,'epochs':np.nan})
            for u,yy,pp in agg(p,te,g,y):oof[n][u]=(yy,pp)
        for j,n in enumerate(['LSTM','BiLSTM','Attn-BiLSTM']):
            t=time.perf_counter(); p,a,k,ep=deep(X,y,g,tr,te,n,SEED+fold*101+j); diag.append({'fold':fold,'model':n,'fit_seconds':time.perf_counter()-t,'epochs':ep}); par.append({'fold':fold,'model':n,'parameters':k})
            for u,yy,pp in agg(p,te,g,y):oof[n][u]=(yy,pp)
            if n=='Attn-BiLSTM':
                for q,ix in enumerate(te):att.append({'subject':g[ix],'label':int(y[ix]),'fold':fold,'window':int(ix%6),'prob_window':float(p[q]),'attention':a[q].tolist(),'raw_flow':raw[ix].tolist()})
    P=[]; M=[]
    for n in names:
        d=pd.DataFrame([(u,*oof[n][u]) for u in us],columns=['subject','y_true','probability']); d['model']=n; P.append(d); yy=d.y_true.to_numpy(); pp=d.probability.to_numpy(); z=metrics(yy,pp)
        for k in ['auc','ap','accuracy','f1']:lo,hi=boot(yy,pp,k);z[k+'_lo']=lo;z[k+'_hi']=hi
        z['model']=n;M.append(z)
    P=pd.concat(P,ignore_index=True); M=pd.DataFrame(M); P.to_csv(OUT/'oof_subject_predictions.csv',index=False); M.to_csv(OUT/'model_metrics.csv',index=False); pd.DataFrame(diag).to_csv(OUT/'training_diagnostics.csv',index=False);pd.DataFrame(par).to_csv(OUT/'deep_model_parameters.csv',index=False);json.dump(att,open(OUT/'attention_records.json','w'))
    ours=P[P.model=='Attn-BiLSTM'].sort_values('subject'); yy=ours.y_true.to_numpy(); po=ours.probability.to_numpy(); C=[]
    for n in names:
        if n=='Attn-BiLSTM':continue
        q=P[P.model==n].sort_values('subject').probability.to_numpy(); a,b,c,p=diff(yy,po,q);C.append({'comparison':f'Attn-BiLSTM - {n}','delta_auc':a,'ci_lo':b,'ci_hi':c,'bootstrap_p':p})
    pd.DataFrame(C).to_csv(OUT/'paired_auc_comparisons.csv',index=False)
    fig,ax=plt.subplots(1,2,figsize=(10,4.2))
    for n in names:
        d=P[P.model==n].sort_values('subject'); yy=d.y_true.to_numpy();pp=d.probability.to_numpy();f,t,_=roc_curve(yy,pp);ax[0].plot(f,t,lw=1.5,label=f'{n} ({roc_auc_score(yy,pp):.3f})'); pr,rc,_=precision_recall_curve(yy,pp);ax[1].plot(rc,pr,lw=1.5,label=f'{n} ({average_precision_score(yy,pp):.3f})')
    ax[0].plot([0,1],[0,1],'--',lw=1);ax[1].axhline(yy.mean(),ls='--',lw=1)
    ax[0].set(xlabel='False-positive rate',ylabel='True-positive rate',title='Subject-level ROC');ax[1].set(xlabel='Recall',ylabel='Precision',title='Subject-level precision–recall')
    for a in ax:a.grid(alpha=.2);a.legend(fontsize=7,frameon=False)
    fig.tight_layout();fig.savefig(OUT/'fig_real_pd_roc_pr.pdf',bbox_inches='tight');fig.savefig(OUT/'fig_real_pd_roc_pr.png',dpi=300,bbox_inches='tight');plt.close(fig)
    O=M.sort_values('auc').reset_index(drop=True);fig,ax=plt.subplots(1,2,figsize=(10.2,4.4));v=O.auc.to_numpy();lo=O.auc_lo.to_numpy();hi=O.auc_hi.to_numpy();q=np.arange(len(O));ax[0].errorbar(v,q,xerr=np.vstack([v-lo,hi-v]),fmt='o',capsize=3);ax[0].set_yticks(q,O.model);ax[0].set(xlabel='ROC-AUC (participant bootstrap 95% CI)',title='Discrimination uncertainty');ax[0].grid(alpha=.2)
    for n in M.sort_values('auc',ascending=False).head(4).model:
        d=P[P.model==n].sort_values('subject');yy=d.y_true.to_numpy();pp=d.probability.to_numpy();fr,mp=calibration_curve(yy,pp,n_bins=6,strategy='quantile');ax[1].plot(mp,fr,marker='o',label=f'{n} (Brier {brier_score_loss(yy,pp):.3f})')
    ax[1].plot([0,1],[0,1],'--');ax[1].set(xlabel='Mean predicted probability',ylabel='Observed PD fraction',title='Calibration of leading models');ax[1].grid(alpha=.2);ax[1].legend(fontsize=7,frameon=False);fig.tight_layout();fig.savefig(OUT/'fig_real_pd_uncertainty_calibration.pdf',bbox_inches='tight');fig.savefig(OUT/'fig_real_pd_uncertainty_calibration.png',dpi=300,bbox_inches='tight');plt.close(fig)
    om={r['subject']:r for r in ours.to_dict('records')};ok=sorted([(u,r['probability']) for u,r in om.items() if r['y_true']==1 and r['probability']>=.5],key=lambda x:x[1])
    if ok:
        u=ok[len(ok)//2][0]; rec=max([r for r in att if r['subject']==u],key=lambda r:r['prob_window']);a=np.asarray(rec['attention']);x=np.asarray(rec['raw_flow']);fig,ax=plt.subplots(2,1,figsize=(9.5,5.2));ax[0].plot(np.arange(len(x))/FS,x,lw=.8);ax[0].set(xlabel='Time in held-out 5-min window (s)',ylabel='z-scored flow',title=f'Representative held-out PD subject {u}');ax[0].grid(alpha=.2);ax[1].plot((np.arange(len(a))+.5)*5,a,marker='o',ms=2.5);ax[1].set(xlabel='Time (s)',ylabel='Attention weight',title='Attention-BiLSTM temporal weighting (5-s frames)');ax[1].grid(alpha=.2);fig.tight_layout();fig.savefig(OUT/'fig_real_pd_attention.pdf',bbox_inches='tight');fig.savefig(OUT/'fig_real_pd_attention.png',dpi=300,bbox_inches='tight');plt.close(fig)
    S={'n_subjects':int(len(us)),'n_pd':int(sy.sum()),'n_control':int((sy==0).sum()),'n_windows':int(len(y)),'sequence_shape':list(X.shape[1:]),'minutes_per_subject':30,'fs_hz':6.0,'seed':SEED};json.dump(S,open(OUT/'experiment_summary.json','w'),indent=2)
    print('\nMETRICS\n',M.to_string(index=False),flush=True);print('\nPAIRWISE\n',pd.DataFrame(C).to_string(index=False),flush=True);print('\nSUMMARY\n',json.dumps(S,indent=2),flush=True)
if __name__=='__main__':main()
