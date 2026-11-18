"""Extend the original leave-one-gateway-out test with matched-height ensembles."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from hybrid_pathloss import load_measurements, packet_groups
from revision_experiments import fit_config


def main():
    root=Path(__file__).resolve().parent
    output=root/'results/gateway_matched_controls';output.mkdir(parents=True,exist_ok=True)
    meta={'sha256':{f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in
                   ['gateway_matched_controls.py','revision_experiments.py','hybrid_pathloss.py','data/samples.csv','data/gateways.csv']},
          'jobs':4,'seed':0,'configs':['direct_height','oh_height'],
          'validation':'15% of remaining packet groups, identical to original gateway study'}
    path=output/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=meta:raise RuntimeError('Provenance changed')
    path.write_text(json.dumps(meta,indent=2)+'\n')
    df,_=load_measurements(root/'data/samples.csv',root/'data/gateways.csv')
    indices=np.arange(len(df));groups=packet_groups(df)
    for gw in sorted(df.gw.unique()):
        test=indices[df.gw.eq(gw)];remainder=indices[df.gw.ne(gw)]
        splitter=GroupShuffleSplit(n_splits=1,test_size=.15,random_state=0)
        tr,va=next(splitter.split(remainder,groups=groups.iloc[remainder]))
        train,val=remainder[tr],remainder[va]
        for config in ['direct_height','oh_height']:
            path=output/f'gateway_{gw}_{config}.json'
            if path.exists():continue
            records,predictions=fit_config(df,(train,val,test),config,0,4)
            pd.DataFrame({'source_row':df.iloc[test].index,'measured_db':df.iloc[test].path_loss.to_numpy(),
                          **{k+'_db':v for k,v in predictions.items()}}).to_csv(path.with_suffix('.csv.gz'),index=False)
            result={'gateway':gw,'seed':0,'config':config,'sizes':{'train':len(train),'validation':len(val),'test':len(test)},'scores':records}
            path.write_text(json.dumps(result,indent=2)+'\n')
            print(gw,config,records[-1]['test_rmse_db'],flush=True)
    rows=[]
    for path in output.glob('gateway*.json'):
        r=json.loads(path.read_text())
        rows.extend({'gateway':r['gateway'],'config':r['config'],'test_rows':r['sizes']['test'],**s} for s in r['scores'])
    pd.DataFrame(rows).to_csv(output/'metrics.csv',index=False)

if __name__=='__main__':main()
