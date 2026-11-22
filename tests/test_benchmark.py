import json
import numpy as np
import pandas as pd
import pytest
import torch
from finforecast.data import synthetic_prices, load_prices, prepare_data, feature_matrix
from finforecast.models import LSTM, TCN, Transformer
from finforecast.benchmark import Settings, score, run_benchmark
from finforecast.cli import main

def test_split_chronology_and_targets():
    d=prepare_data(synthetic_prices(250,seed=2),window=12)
    assert max(d.indices['train'])<min(d.indices['val'])
    assert max(d.indices['val'])<min(d.indices['test'])
    for i in [0,len(d.targets)//2,len(d.targets)-1]:
        price_pos=d.prices.index.get_loc(d.target_dates[i])
        expected=np.log(d.prices.close.iloc[price_pos]/d.prices.close.iloc[price_pos-1])
        assert np.isclose(d.targets[i],expected,atol=1e-7)

def test_future_prices_do_not_change_training_features():
    prices=synthetic_prices(270)
    changed=prices.copy()
    changed.iloc[230:,0]*=3
    a=prepare_data(prices,window=16)
    b=prepare_data(changed,window=16)
    assert np.array_equal(a.mean,b.mean)
    assert np.array_equal(a.std,b.std)
    assert np.array_equal(a.windows[a.indices['train']],b.windows[b.indices['train']])

def test_future_returns_do_not_change_earlier_feature_rows():
    p=synthetic_prices(180)
    q=p.copy()
    q.iloc[135:,0]*=10
    x,_=feature_matrix(p)
    y,_=feature_matrix(q)
    assert np.allclose(x[:134],y[:134],equal_nan=True)

@pytest.mark.parametrize('model',[LSTM,TCN,Transformer])
def test_model_output_shapes(model):
    torch.manual_seed(0)
    m=model(features=3,hidden=8)
    x=torch.randn(4,12,3)
    assert m(x).shape==(4,)
    assert torch.isfinite(m(x)).all()

def test_model_can_be_trained_and_all_baselines_evaluated():
    d=prepare_data(synthetic_prices(260),window=12)
    report,table=run_benchmark(d,Settings(epochs=1,hidden=8,batch_size=128))
    assert set(report)=={'Zero forecast','Ridge','ARIMA(1,0,0)','LSTM','TCN','Transformer'}
    assert len(table)==(len(d.indices['val'])+len(d.indices['test']))*6
    assert all(np.isfinite(v['test']['mse']) for v in report.values())

def test_csv_validation(tmp_path):
    f=tmp_path/'bad.csv'
    f.write_text('date,close\n2023-01-01,100\n2023-01-01,101\n')
    with pytest.raises(ValueError):load_prices(f)

def test_undersized_and_invalid_settings():
    with pytest.raises(ValueError):synthetic_prices(20)
    with pytest.raises(ValueError):prepare_data(synthetic_prices(180),window=1)
    with pytest.raises(ValueError):score(np.array([0.]),np.array([float('nan')]))

def test_zero_forecast_cost_and_direction():
    result=score([-.01,.02,-.01],[0.,0.,0.])
    assert result['turnover']==0 and result['total_net_return']==0

def test_reproducible_training():
    d=prepare_data(synthetic_prices(220),window=10)
    cfg=Settings(epochs=1,hidden=8,batch_size=128,seed=99)
    a,_=run_benchmark(d,cfg)
    b,_=run_benchmark(d,cfg)
    for name in a:
        assert a[name]['test']['mse']==b[name]['test']['mse']

def test_cli_writes_artifacts(tmp_path):
    main(['--days','220','--window','12','--epochs','1','--hidden','8','--output',str(tmp_path)])
    assert (tmp_path/'predictions.csv').exists()
    assert (tmp_path/'metrics.json').exists()
    assert (tmp_path/'test_mse.png').exists()
    assert len(json.loads((tmp_path/'metrics.json').read_text())['models'])==6
