import numpy as np
import pandas as pd
import mlflow
from hyperopt import hp, fmin, tpe

from spreading_activation import Spreader


def log_cfg(cfg):
    for key, value in cfg.items():
        if type(value) is dict:
            log_cfg(value)
        else:
            mlflow.log_param(str(key).split('/')[-1], value)


spreader = Spreader()
data = pd.read_csv('fold_1/t.csv')
val_data = pd.read_csv('fold_1/val.csv')
uids = [30, 112, 234, 1111, 1234]  # change these maybe? idk


def rate_for_uid(uid, k=50):
    movies_to_activate = data[(data['UID'] == uid) & (data['rating'] >= 3)]['OID']
    spreader.spread(movies_to_activate)
    top_k = spreader.get_top_k_as_list(k)
    recs = np.array([int(rec[1]) for rec in top_k])
    run_ndcg = ndcg(uid, recs, top_k=k)
    return run_ndcg


def spread_and_rate(cfg):
    with mlflow.start_run():
        log_cfg(cfg)
        spreader.update_cfg(cfg)
        total_ndcg = 0
        for uid in uids:
            total_ndcg += rate_for_uid(uid)
        total_ndcg /= len(uids)
        mlflow.log_metric('ndcg', total_ndcg)
    return 1 - total_ndcg


def main():
    cfg_space = {
        'activation_threshold': hp.uniform('activation_threshold', 0.3, 0.75),  # Min activation a node needs to spread
        'decay_factor': hp.uniform('decay_factor', 0.01, 0.2),  # What part of the activation will survive the spread
        'edge_weights': {  # Proportional amount of activation that will flow along the edge
            'https://example.org/fromDecade': hp.uniform('fromDecade', 0.1, 0.7),
            'https://schema.org/Actor': hp.uniform('Actor', 0.2, 0.8),
            'https://schema.org/Director': hp.uniform('Director', 0.05, 0.4),
            'https://schema.org/Writer': hp.uniform('Writer', 0.3, 0.9),
            'https://schema.org/Producer': hp.uniform('Producer', 0.1, 0.8),
            'https://schema.org/Editor': hp.uniform('Editor', 0.1, 0.4),
            'https://schema.org/Genre': hp.uniform('Genre', 0.1, 0.6),
            'https://schema.org/CountryOfOrigin': hp.uniform('CountryOfOrigin', 0.1, 0.7),
            'https://schema.org/inLanguage': hp.uniform('inLanguage', 0.4, 0.8),
        }
    }
    best = fmin(fn=spread_and_rate, space=cfg_space, algo=tpe.suggest, max_evals=100)
    print(best)


def ndcg(uid, recs, top_k=20):
    user_data = val_data.loc[val_data.UID == uid]
    recs = recs[:top_k]
    dcg_penalty = 1 / np.log2(np.array(range(top_k)) + 2)

    ideal = user_data.rating.sort_values(ascending=False).values[:top_k]
    if len(ideal) > 0:
        # print(ideal, len(ideal), np.array(dcg_penalty[:len(ideal)]) )
        idcg = (ideal * dcg_penalty[:len(ideal)]).sum()
        user_data.set_index("OID", inplace=True)
        recs_relevant = [user_data.rating[i] if i in user_data.index else 0 for i in recs]
        dcg = (np.array(recs_relevant) * dcg_penalty).sum()
        return dcg / idcg
    else:
        return 0


if __name__ == "__main__":
    main()
