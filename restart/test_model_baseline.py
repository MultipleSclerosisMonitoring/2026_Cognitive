"""Synthetic checks of the critical validation boundaries; no clinical records."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedGroupKFold
from restart.model_baseline import pipeline, BASE, discrimination


def synthetic():
    rng=np.random.default_rng(51)
    n=60
    return pd.DataFrame({'pid':np.arange(n),'group':np.repeat([0,1],30),
        'age':rng.uniform(25,65,n),'sex':np.tile(['hombre','mujer'],30),
        'education':np.tile(['básicos','medios','superiores'],20),'y':rng.normal(50,10,n)})


def test_preprocessing_uses_training_only():
    d=synthetic(); train=d.iloc[:40].copy(); test=d.iloc[40:].copy()
    train.loc[0,'age']=np.nan
    model=pipeline(BASE,Ridge(alpha=10)).fit(train[BASE],train.y)
    median=model['prep'].named_transformers_['num']['impute'].statistics_[0]
    assert median==pytest.approx(train.age.median())
    test['age']=10000
    model.predict(test[BASE])
    assert model['prep'].named_transformers_['num']['impute'].statistics_[0]==median


def test_bootstrap_copies_cannot_cross_folds():
    d=pd.concat([synthetic(),synthetic().iloc[:10]],ignore_index=True)
    for tr,te in StratifiedGroupKFold(5,shuffle=True,random_state=123).split(d,d.group,d.pid):
        assert set(d.pid.iloc[tr]).isdisjoint(d.pid.iloc[te])


def test_discrimination_reproducible_with_missing_predictor():
    d=synthetic(); d.loc[0,'age']=np.nan
    a=discrimination(d); b=discrimination(d)
    assert a==b
    for name in ['demographic','demographic_cognitive','normative_deviation']:
        assert 0<=a[name]['auc']<=1
