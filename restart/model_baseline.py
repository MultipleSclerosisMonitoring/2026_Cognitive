"""Exploratory baseline models, aggregate output only; one baseline row per person."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import xlrd
import statsmodels.formula.api as smf
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error, r2_score, balanced_accuracy_score, confusion_matrix, brier_score_loss
from statsmodels.stats.multitest import multipletests
from restart.audit import audit

BASE=['age','sex','education']
SEED=20260930


def load(path):
    check=audit(path)
    for group in check['groups']:
        for name,v in group['variables'].items():
            if v['non_numeric_or_nonfinite'] or (name=='edad' and v['nonpositive_numeric']):
                raise ValueError('Resolve invalid baseline values before modeling')
    book=xlrd.open_workbook(str(path)); records=[]; seen=set()
    for si,h,age,sdmt,ta,tb,tug in [(0,2,12,36,30,31,22),(1,1,10,30,24,25,16)]:
        s=book.sheet_by_index(si)
        if 'TUG' not in str(s.cell_value(h,tug)): raise ValueError('Unexpected TUG column')
        for i in range(h+1,s.nrows):
            code=str(s.cell_value(i,0)).strip()
            if not code:
                if any(s.cell_value(i,c)!='' for c in [sdmt,ta,tb]):
                    raise ValueError('Uncoded row with outcome')
                continue
            if code in seen: raise ValueError('Duplicate participant code')
            seen.add(code)
            r={'pid':len(records),'group':1-si}
            for name,col in [('age',age),('sdmt',sdmt),('tmt_a',ta),('tmt_b',tb),('tug',tug)]:
                v=s.cell_value(i,col)
                if v=='': r[name]=np.nan
                elif isinstance(v,(float,int)) and np.isfinite(v): r[name]=float(v)
                else: raise ValueError(f'Unexpected numeric value in {name}')
            for name,col,allowed in [('sex',4,{'hombre','mujer'}),('education',5,{'básicos','medios','superiores'})]:
                v=str(s.cell_value(i,col)).strip().lower()
                if v and v not in allowed: raise ValueError(f'Unexpected category in {name}')
                r[name]=v if v else np.nan
            for name in ['age','tmt_a','tmt_b','tug']:
                if pd.notna(r[name]) and r[name]<=0: raise ValueError(f'Nonpositive {name}')
            if pd.notna(r['sdmt']) and r['sdmt']<0: raise ValueError('Negative SDMT')
            records.append(r)
    return pd.DataFrame(records),check


def pipeline(features, estimator):
    numeric=[c for c in features if c not in ['sex','education']]
    cats=[c for c in features if c in ['sex','education']]
    prep=ColumnTransformer([
        ('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),numeric),
        ('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),
         ('encode',OneHotEncoder(categories=[['hombre','mujer'] if c=='sex' else ['básicos','medios','superiores'] for c in cats],drop='first',handle_unknown='error',sparse_output=False))]),cats)])
    return Pipeline([('prep',prep),('model',estimator)])


def infer(d):
    c=d.dropna(subset=BASE).copy()
    formula='y ~ group + age + C(sex) + C(education)'
    fit=smf.ols(formula,c).fit(cov_type='HC3',use_t=True)
    if np.linalg.matrix_rank(fit.model.exog)<fit.model.exog.shape[1]: raise ValueError('Rank deficient model')
    ci=fit.conf_int().loc['group'].tolist()
    result={'n':len(c),'n_controls':int((c.group==0).sum()),'formula':formula,
        'group_effect':float(fit.params['group']),'group_ci95':ci,'group_p':float(fit.pvalues['group']),
        'coefficients':{k:{'estimate':float(v),'ci95':fit.conf_int().loc[k].tolist(),'p':float(fit.pvalues[k])} for k,v in fit.params.items()},
        'separate_models':{}}
    for g in [0,1]:
        sub=c[c.group==g]
        m=smf.ols('y ~ age + C(sex) + C(education)',sub).fit(cov_type='HC3',use_t=True)
        result['separate_models'][str(g)]={'n':len(sub),'rank':int(np.linalg.matrix_rank(m.model.exog)),
            'parameters':len(m.params),'coefficients':{k:float(v) for k,v in m.params.items()},'ci95':m.conf_int().to_dict(orient='index')}
    interaction=smf.ols('y ~ group * age + C(sex) + C(education)',c).fit(cov_type='HC3',use_t=True)
    result['age_interaction_p']=float(interaction.pvalues['group:age'])
    result['group_and_age_interaction_joint_p']=float(interaction.f_test('group=0, group:age=0').pvalue)
    a=c[c.group==0].age; b=c[c.group==1].age
    lo,hi=max(a.min(),b.min()),min(a.max(),b.max())
    overlap=c[c.age.between(lo,hi)]
    m=smf.ols(formula,overlap).fit(cov_type='HC3',use_t=True)
    result['age_overlap_sensitivity']={'range':[lo,hi],'n':len(overlap),'effect':float(m.params['group']),'ci95':m.conf_int().loc['group'].tolist(),'note':'Age overlap only; not full multivariate common support'}
    return result


def regression_cv(d):
    output=[]
    for g in [0,1]:
        sub=d[d.group==g].reset_index(drop=True)
        for name,features,est in [('mean',BASE,None),('linear',BASE,LinearRegression()),('ridge_base',BASE,Ridge()),('ridge_tug',BASE+['tug'],Ridge())]:
            pred=np.full(len(sub),np.nan); alphas=[]
            for tr,te in GroupKFold(5).split(sub,groups=sub.pid):
                if est is None: pred[te]=sub.y.iloc[tr].mean(); continue
                model=pipeline(features,est)
                if name.startswith('ridge'):
                    model=GridSearchCV(model,{'model__alpha':[0.1,1,10,100]},cv=GroupKFold(3),scoring='neg_mean_absolute_error',error_score='raise')
                    model.fit(sub.iloc[tr][features],sub.y.iloc[tr],groups=sub.pid.iloc[tr]); alphas.append(float(model.best_params_['model__alpha']))
                else: model.fit(sub.iloc[tr][features],sub.y.iloc[tr])
                pred[te]=model.predict(sub.iloc[te][features])
            output.append({'group':g,'model':name,'n':len(sub),'mae':mean_absolute_error(sub.y,pred),
                'rmse':float(np.sqrt(mean_squared_error(sub.y,pred))),'r2':r2_score(sub.y,pred),'selected_alphas':alphas})
    return output


def discrimination(d,seed=SEED):
    """Fixed hyperparameters; all transforms/norm fits are within each outer fold."""
    pred={name:np.zeros(len(d)) for name in ['demographic','demographic_cognitive','normative_deviation']}
    folds=StratifiedGroupKFold(5,shuffle=True,random_state=seed)
    for tr,te in folds.split(d,d.group,d.pid):
        train,test=d.iloc[tr],d.iloc[te]
        assert not set(train.pid)&set(test.pid)
        for name,features in [('demographic',BASE),('demographic_cognitive',BASE+['y'])]:
            model=pipeline(features,LogisticRegression(C=1,max_iter=2000))
            model.fit(train[features],train.group)
            pred[name][te]=model.predict_proba(test[features])[:,1]
        controls=train[train.group==0]
        model=pipeline(BASE,Ridge(alpha=10))
        model.fit(controls[BASE],controls.y)
        deviation=model.predict(test[BASE])-test.y.to_numpy()
        # y is SDMT or negative log(TMT), hence positive deviation always means worse.
        pred['normative_deviation'][te]=deviation
    result={}
    for name,p in pred.items():
        result[name]={'auc':float(roc_auc_score(d.group,p))}
        if name!='normative_deviation':
            tn,fp,fn,tp=confusion_matrix(d.group,p>=0.5,labels=[0,1]).ravel()
            result[name].update({'sensitivity':float(tp/(tp+fn)),'specificity':float(tn/(tn+fp)),
                'balanced_accuracy':float(balanced_accuracy_score(d.group,p>=.5)),
                'brier':float(brier_score_loss(d.group,p)),'threshold':0.5})
    result['auc_increment']=result['demographic_cognitive']['auc']-result['demographic']['auc']
    return result


def bootstrap_discrimination(d,n):
    rng=np.random.default_rng(SEED); values=[]; failures=0
    for iteration in range(n):
        idx=np.concatenate([rng.choice(np.flatnonzero(d.group.to_numpy()==g),sum(d.group==g),replace=True) for g in [0,1]])
        sample=d.iloc[idx].reset_index(drop=True)
        if min(sample.groupby('group').pid.nunique())<5: failures+=1; continue
        try:
            r=discrimination(sample,SEED+iteration)
            values.append([r[k]['auc'] for k in ['demographic','demographic_cognitive','normative_deviation']]+[r['auc_increment']])
        except (ValueError,np.linalg.LinAlgError): failures+=1
    if not values: raise ValueError('No valid bootstrap replicates')
    return {'requested':n,'successful':len(values),'failed':failures,'method':'Stratified participant bootstrap; duplicate copies stay in same fold; full refit; percentile intervals',
        'ci95':dict(zip(['demographic_auc','demographic_cognitive_auc','normative_deviation_auc','auc_increment'],np.quantile(values,[.025,.975],axis=0).T.tolist()))}


def power_grid(d):
    # Conditional-design Monte Carlo, fixed covariates, robust HC3 test of group.
    c=d.dropna(subset=BASE)
    fitted=smf.ols('y ~ group + age + C(sex) + C(education)',c).fit()
    x=fitted.model.exog; gi=fitted.model.exog_names.index('group')
    pinv=np.linalg.pinv(x); leverage=np.sum(x*pinv.T,axis=1)
    rng=np.random.default_rng(SEED); out=[]; reps=2000
    for sigma in [8,12,16]:
        for ratio in [1,1.5]:
            noise=rng.normal(size=(len(c),reps))*np.where(c.group.to_numpy()[:,None]==1,sigma*ratio,sigma)
            for effect in [0,3,5,8,10]:
                y=noise-effect*c.group.to_numpy()[:,None]
                beta=pinv@y; residual=y-x@beta
                se=np.sqrt(np.sum((pinv[gi,:,None]*residual/(1-leverage[:,None]))**2,axis=0))
                # Same finite-df t reference as HC3 inferential fit.
                from scipy.stats import t
                reject=np.abs(beta[gi]/se)>t.ppf(.975,len(c)-x.shape[1])
                p=float(reject.mean())
                out.append({'sd_controls':sigma,'sd_patient_ratio':ratio,'difference_points':effect,'power_or_type1_if_zero':p,'mc_se':float(np.sqrt(p*(1-p)/reps)),'replicates':reps})
    return {'assumptions':'Fixed complete-case covariate design; Gaussian errors; specified SD/effects, no selection; no missingness simulation or sample-size expansion. Planning scenarios, not observed power.','scenarios':out}


def main(path,out,boot):
    out.mkdir(parents=True,exist_ok=True)
    frame,audit_report=load(path)
    result={'audit':audit_report,'seed':SEED,'bootstrap_replicates':boot,
        'versions':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','statsmodels','xlrd']},
        'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'outcomes':{}}
    for outcome in ['sdmt','tmt_a','tmt_b']:
        print(f'Analyzing {outcome}',flush=True)
        d=frame.dropna(subset=[outcome]).copy().reset_index(drop=True)
        d['y']=d[outcome] if outcome=='sdmt' else -np.log(d[outcome])
        r={'scale':'points' if outcome=='sdmt' else 'negative log seconds',
            'n':len(d),'n_by_group':d.groupby('group').size().to_dict(),
            'descriptive':d.groupby('group')[outcome].agg(['count','mean','std','median']).to_dict(orient='index'),
            'missing_predictors':d[BASE+['tug']].isna().sum().to_dict(),
            'inference':infer(d),'regression_cv':regression_cv(d),
            'discrimination':discrimination(d),'bootstrap':bootstrap_discrimination(d,boot)}
        r['split_stability']=[discrimination(d,SEED+i) for i in [1,2]]
        if outcome=='sdmt': r['power']=power_grid(d)
        else:
            r['inference']['patient_control_time_ratio']=float(np.exp(-r['inference']['group_effect']))
            r['inference']['time_ratio_ci95']=np.exp(-np.array(r['inference']['group_ci95'])[::-1]).tolist()
        result['outcomes'][outcome]=r
        (out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    corrected=multipletests([result['outcomes'][k]['inference']['group_p'] for k in ['tmt_a','tmt_b']],method='holm')[1]
    for key,p in zip(['tmt_a','tmt_b'],corrected): result['outcomes'][key]['inference']['group_p_holm']=float(p)
    (out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print('Finished',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path); parser.add_argument('--output',type=Path,default=Path('reports/baseline'))
    parser.add_argument('--bootstrap',type=int,default=200)
    args=parser.parse_args(); main(args.input,args.output,args.bootstrap)
