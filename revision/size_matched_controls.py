"""Control training/validation sample count separately from spatial exclusion."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from hybrid_pathloss import load_measurements, split_rows
from revision_experiments import buffered_split, fit_config


def main():
    root=Path(__file__).resolve().parent
    output=root/'results/size_matched_controls';output.mkdir(parents=True,exist_ok=True)
    meta={'sha256':{f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in
                   ['size_matched_controls.py','revision_experiments.py','hybrid_pathloss.py','data/samples.csv','data/gateways.csv']},
          'jobs':4,'seeds':list(range(5)),'configs':['direct_height','oh_height'],
          'design':'Same original test tiles; random subsets of original training and validation rows equal in size to buffered retained sets. No proximity exclusion.'}
    manifest=output/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=meta:raise RuntimeError('Provenance changed')
    manifest.write_text(json.dumps(meta,indent=2)+'\n')
    df,_=load_measurements(root/'data/samples.csv',root/'data/gateways.csv')
    for radius in [250,500,1000]:
        for seed in range(5):
            _,sizes=buffered_split(df,seed,radius)
            if sizes['train']<100 or sizes['validation']<100:continue
            train,val,test=split_rows(df,'spatial_500m',seed)
            rng=np.random.default_rng(seed)
            train=np.sort(rng.choice(train,sizes['train'],replace=False))
            val=np.sort(rng.choice(val,sizes['validation'],replace=False))
            for config in ['direct_height','oh_height']:
                path=output/f'size{radius}_seed{seed}_{config}.json'
                if path.exists():continue
                records,pred=fit_config(df,(train,val,test),config,seed,4)
                pd.DataFrame({'source_row':df.iloc[test].index,'measured_db':df.iloc[test].path_loss.to_numpy(),
                              **{k+'_db':v for k,v in pred.items()}}).to_csv(path.with_suffix('.csv.gz'),index=False)
                path.write_text(json.dumps({'radius':radius,'seed':seed,'config':config,'sizes':sizes,'scores':records},indent=2)+'\n')
                print(radius,seed,config,records[-1]['test_rmse_db'],flush=True)
    rows=[]
    for path in output.glob('size*.json'):
        r=json.loads(path.read_text())
        rows.extend({'radius':r['radius'],'seed':r['seed'],'config':r['config'],**v} for v in r['scores'])
    pd.DataFrame(rows).to_csv(output/'metrics.csv',index=False)

if __name__=='__main__':main()
