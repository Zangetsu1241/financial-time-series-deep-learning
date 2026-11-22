"""Causal features, chronological splits, and training-only standardization."""
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class Dataset:
    windows: np.ndarray
    targets: np.ndarray
    target_dates: pd.DatetimeIndex
    indices: dict
    mean: np.ndarray
    std: np.ndarray
    returns: np.ndarray
    prices: pd.DataFrame
    window_size: int

def synthetic_prices(n=900,seed=42):
    if n<150:raise ValueError("at least 150 trading days needed")
    rng=np.random.default_rng(seed)
    innovations=rng.normal(size=n)
    returns=np.zeros(n)
    state=0
    for t in range(1,n):
        if rng.random()<.014:state=1-state
        scale=.007 if state==0 else .022
        returns[t]=.18*returns[t-1]+.00015+scale*innovations[t]
    return pd.DataFrame({"close":100*np.exp(np.cumsum(returns))},
                        index=pd.bdate_range("2018-01-01",periods=n))

def load_prices(path):
    df=pd.read_csv(path)
    if not {"date","close"}.issubset(df.columns):
        raise ValueError("CSV requires date,close")
    dates=pd.to_datetime(df["date"],utc=True,errors="raise")
    close=pd.to_numeric(df["close"],errors="raise").to_numpy(float)
    if (dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing
        or not np.isfinite(close).all() or np.any(close<=0)):
        raise ValueError("dates must be ordered unique, closes positive and finite")
    if len(close)<150:raise ValueError("not enough price history")
    return pd.DataFrame({"close":close},index=dates)

def feature_matrix(prices):
    close=prices["close"].to_numpy(dtype=float)
    if len(close)<150 or not np.isfinite(close).all() or np.any(close<=0):
        raise ValueError("need >=150 positive finite closes")
    r=np.diff(np.log(close),prepend=np.nan)
    s=pd.Series(r)
    vol=s.rolling(10,min_periods=10).std().to_numpy()
    momentum=s.rolling(5,min_periods=5).sum().to_numpy()
    return np.column_stack([r,vol,momentum]),r

def prepare_data(prices,window=24,train_fraction=.65,val_fraction=.15):
    if window<4 or not (.4<=train_fraction<=.8) or not (.05<=val_fraction<=.3) or train_fraction+val_fraction>=.9:
        raise ValueError("invalid split/window settings")
    if not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError("price dates must be strictly ordered")
    x,r=feature_matrix(prices)
    n=len(r)
    train_end=int(n*train_fraction)
    val_end=int(n*(train_fraction+val_fraction))
    first=max(11,window-1)
    if train_end-first<25 or val_end-train_end<10 or n-val_end<10:
        raise ValueError("insufficient history for splits")
    decisions=np.arange(first,n-1)
    decisions=decisions[np.array([np.isfinite(x[t-window+1:t+1]).all() for t in decisions])]
    if len(decisions)==0:raise ValueError("no usable training samples")
    historic=x[np.isfinite(x).all(axis=1)&(np.arange(n)<=train_end-2)]
    mean=historic.mean(axis=0)
    std=historic.std(axis=0)
    std=np.where(std<1e-10,1.,std)
    windows=np.stack([(x[t-window+1:t+1]-mean)/std for t in decisions]).astype("float32")
    targets=r[decisions+1].astype("float32")
    positions=decisions+1
    indices={"train":np.where(positions<train_end)[0],
             "val":np.where((positions>=train_end)&(positions<val_end))[0],
             "test":np.where(positions>=val_end)[0]}
    if min(map(len,indices.values()))<10:raise ValueError("empty or undersized split")
    return Dataset(windows,targets,prices.index[positions],indices,mean,std,r,prices,window)
