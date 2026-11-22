"""CLI for reproducible financial forecasting experiments."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .data import synthetic_prices,load_prices,prepare_data
from .benchmark import Settings,run_benchmark

def main(argv=None):
    p=argparse.ArgumentParser(description="Financial return prediction model comparison")
    p.add_argument("--csv",default=None,help="date,close with positive adjusted daily prices")
    p.add_argument("--days",type=int,default=900)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--window",type=int,default=24)
    p.add_argument("--epochs",type=int,default=8)
    p.add_argument("--hidden",type=int,default=24)
    p.add_argument("--cost-bps",type=float,default=5.)
    p.add_argument("--output",default="results")
    args=p.parse_args(argv)
    prices=load_prices(args.csv) if args.csv else synthetic_prices(args.days,args.seed)
    data=prepare_data(prices,window=args.window)
    cfg=Settings(epochs=args.epochs,hidden=args.hidden,seed=args.seed,cost_bps=args.cost_bps)
    report,obs=run_benchmark(data,cfg)
    dest=Path(args.output)
    dest.mkdir(parents=True,exist_ok=True)
    obs.to_csv(dest/"predictions.csv",index=False)
    summary={"dataset":"user-supplied CSV" if args.csv else "synthetic with regime-dependent volatility",
             "config":vars(args),"splits":{k:len(v) for k,v in data.indices.items()},
             "research_limitations":"Close-t information predicts next close-to-close returns; fills at the known close after observing it are impossible. Trading metrics are illustrative.",
             "models":report}
    (dest/"metrics.json").write_text(json.dumps(summary,indent=2))
    labels=list(report)
    values=[report[k]["test"]["mse"] for k in labels]
    fig,ax=plt.subplots(figsize=(10,4))
    ax.bar(labels,values)
    ax.set_ylabel("Test mean squared log return error")
    ax.tick_params(axis="x",rotation=30)
    fig.tight_layout()
    fig.savefig(dest/"test_mse.png",dpi=160)
    plt.close(fig)
    print(json.dumps({k:{"test_mse":v["test"]["mse"],"test_mae":v["test"]["mae"],
                         "test_direction_accuracy":v["test"]["direction_accuracy"]}
                      for k,v in report.items()},indent=2))
    return summary

if __name__=="__main__":main()
