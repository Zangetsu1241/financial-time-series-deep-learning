"""Chronological benchmarks with validation-only checkpoint selection."""
from dataclasses import dataclass
from copy import deepcopy
import random
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error,mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from .models import MODELS

@dataclass(frozen=True)
class Settings:
    epochs:int=8
    batch_size:int=64
    lr:float=.002
    hidden:int=24
    seed:int=42
    threads:int=1
    cost_bps:float=5.0

def score(y,pred,cost_bps=5.0):
    y=np.asarray(y,float)
    pred=np.asarray(pred,float)
    if y.shape!=pred.shape or len(y)==0 or not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError("predictions must be finite and aligned")
    positions=np.sign(pred)
    turnover=np.abs(positions-np.r_[0,positions[:-1]])
    net=positions*y-turnover*cost_bps/10000
    return {"mse":float(mean_squared_error(y,pred)),
            "mae":float(mean_absolute_error(y,pred)),
            "direction_accuracy":float(np.mean(np.sign(y)==np.sign(pred))),
            "mean_net_log_return":float(np.mean(net)),
            "total_net_return":float(np.exp(net.sum())-1),
            "turnover":float(turnover.sum())}

def _fit_network(name,dataset,cfg):
    model=MODELS[name](features=dataset.windows.shape[-1],hidden=cfg.hidden)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg.lr)
    X=torch.tensor(dataset.windows)
    Y=torch.tensor(dataset.targets*100.)
    train_idx=dataset.indices["train"]
    val_idx=dataset.indices["val"]
    best=float("inf")
    checkpoint=deepcopy(model.state_dict())
    rng=np.random.default_rng(cfg.seed)
    for _ in range(cfg.epochs):
        model.train()
        order=rng.permutation(train_idx)
        for j in range(0,len(order),cfg.batch_size):
            ix=order[j:j+cfg.batch_size]
            optimizer.zero_grad()
            loss=torch.nn.functional.mse_loss(model(X[ix]),Y[ix])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val=torch.nn.functional.mse_loss(model(X[val_idx]),Y[val_idx]).item()
        if val<best:
            best=val
            checkpoint=deepcopy(model.state_dict())
    model.load_state_dict(checkpoint)
    model.eval()
    with torch.no_grad():predictions=model(X).numpy()/100.
    return predictions,{"parameters":sum(p.numel() for p in model.parameters()),
                         "best_validation_mse_percent2":best}

def _arima_predictions(data):
    """Train-only ARIMA estimation and sequential one-step held-out forecasts."""
    idx=data.indices
    all_returns=pd.Series(data.returns,index=data.prices.index).dropna()
    last_train_target=data.target_dates[idx["train"][-1]]
    history=all_returns.loc[:last_train_target]
    model=ARIMA(history,order=(1,0,0),trend="c").fit()
    result=np.full(len(data.targets),np.nan)
    target_locations=data.prices.index.get_indexer(data.target_dates)
    previous_index=data.prices.index.get_loc(last_train_target)
    for i in np.r_[idx["val"],idx["test"]]:
        position=target_locations[i]
        while previous_index+1<position:
            previous_index+=1
            model=model.append([float(data.returns[previous_index])],refit=False)
        result[i]=float(np.asarray(model.forecast(steps=1))[0])
        model=model.append([float(data.returns[position])],refit=False)
        previous_index=position
    return result

def run_benchmark(data,cfg=Settings()):
    if cfg.epochs<1 or cfg.batch_size<1 or cfg.hidden<4 or cfg.cost_bps<0 or cfg.threads<1:
        raise ValueError("invalid training configuration")
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.threads)
    idx=data.indices
    X=data.windows.reshape(len(data.windows),-1)
    y=data.targets
    ridge=Ridge(alpha=10.).fit(X[idx["train"]],y[idx["train"]])
    predictions={"Zero forecast":np.zeros_like(y),"Ridge":ridge.predict(X),
                 "ARIMA(1,0,0)":_arima_predictions(data)}
    details={"Zero forecast":{"parameters":0},
             "Ridge":{"parameters":int(X.shape[1]+1)},
             "ARIMA(1,0,0)":{"parameters":3}}
    for name in MODELS:
        predictions[name],details[name]=_fit_network(name,data,cfg)
    entries=[]
    report={}
    for name,pred in predictions.items():
        report[name]={"details":details[name]}
        for split in ["val","test"]:
            ids=idx[split]
            report[name][split]=score(y[ids],pred[ids],cfg.cost_bps)
            for i in ids:
                entries.append({"date":data.target_dates[i],"split":split,"model":name,
                                "actual_log_return":float(y[i]),
                                "predicted_log_return":float(pred[i])})
    return report,pd.DataFrame(entries)
