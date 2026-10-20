import json
import numpy as np
from joblib import Parallel, delayed
from common import *

d = load_data()
Xtr, ytr, Xte, yte, Xm = d["Xtr"], d["ytr"], d["Xte"], d["yte"], d["Xtm"]
frozen = json.load(open(f"{RES}/frozen.json"))
params, B = frozen["tree_params"], frozen["B"]
bd, bT = frozen["boost_depth"], frozen["boost_rounds"]
R = 30

# the selected tree already has no depth limit, so the deepest option is no limit with 1-sample leaves
settings = {
    "shallow": dict(params, max_depth=4),
    "selected": dict(params),
    "deepest": dict(params, max_depth=None, min_samples_leaf=1),
}


def one_refit(r):
    rng = np.random.default_rng(SEED + r + 1)
    idx = np.concatenate([rng.choice(np.where(ytr == c)[0], 6, replace=True) for c in range(K)])
    X, y = Xtr[idx], ytr[idx]
    out = {}
    for name, p in settings.items():
        tree = new_tree(p, rng.integers(2**31 - 1)).fit(X, y)
        n_trees = B if name == "selected" else 100  # Q3(b) uses 100 trees; final B is also 100
        bag = Bagging(n_trees, p, rng.integers(2**31 - 1)).fit(X, y)
        out[name, "leaf"] = leaf_stats(tree, X)
        out[name, "disagree"] = pairwise_disagreement(bag.tree_preds(Xte))
        for ts, Xs in [("clean", Xte), ("masked", Xm)]:
            out[name, "tree", ts] = tree.predict_proba(Xs)
            out[name, "bag", ts] = bag.vote_frac(Xs)
    boost = SAMME(bd, bT, rng.integers(2**31 - 1)).fit(X, y)
    for ts, Xs in [("clean", Xte), ("masked", Xm)]:
        out["boost", ts] = boost.vote(Xs)
    print(f"refit {r + 1} done", flush=True)
    return out


def bias_variance(P):
    # P has shape (R, N, K): class vector for every refit and test face
    p_mean = P.mean(0)
    target = np.eye(K)[yte]
    bias2 = ((p_mean - target) ** 2).sum(1).mean()
    var = ((P - p_mean) ** 2).sum(2).mean()
    accs = (P.argmax(2) == yte).mean(1)
    return dict(bias2=float(bias2), variance=float(var), acc_mean=float(accs.mean()), acc_std=float(accs.std()))


runs = Parallel(n_jobs=4)(delayed(one_refit)(r) for r in range(R))
get = lambda key: np.stack([run[key] for run in runs])


def summary(key):
    s = bias_variance(get(key + ("clean",)))
    s["masked_acc_mean"] = float((get(key + ("masked",)).argmax(2) == yte).mean())
    return s


# (a) selected models
print("== Q3(a) selected models, 30 refits ==")
main = {"tree": summary(("selected", "tree")), "bagging": summary(("selected", "bag")), "boosting": summary(("boost",))}
for m, s in main.items():
    print(f"{m:9s} bias2={s['bias2']:.4f} variance={s['variance']:.4f} "
          f"test acc={s['acc_mean']:.4f} +- {s['acc_std']:.4f}")

# (b) and (c) three depths, single tree vs 100-tree bagging
print("== Q3(b,c) depth study ==")
depth_rows = {}
for name in settings:
    t, b = summary((name, "tree")), summary((name, "bag"))
    leaf = {k: float(np.mean([run[name, "leaf"][k] for run in runs])) for k in runs[0][name, "leaf"]}
    row = dict(tree=t, bag=b, leaf=leaf,
               disagreement=float(np.mean([run[name, "disagree"] for run in runs])),
               variance_removed=t["variance"] - b["variance"],
               variance_removed_pct=100 * (t["variance"] - b["variance"]) / t["variance"],
               acc_gain=b["acc_mean"] - t["acc_mean"])
    depth_rows[name] = row
    print(f"{name:8s} depth~{leaf['depth']:.1f} leaves={leaf['n_leaves']:.1f} "
          f"samples/leaf={leaf['mean_samples_per_leaf']:.2f} singleton={leaf['frac_singleton_leaves']:.3f}")
    print(f"   tree: bias2={t['bias2']:.4f} var={t['variance']:.4f} acc={t['acc_mean']:.4f}")
    print(f"   bag : bias2={b['bias2']:.4f} var={b['variance']:.4f} acc={b['acc_mean']:.4f}")
    print(f"   variance removed={row['variance_removed']:.4f} ({row['variance_removed_pct']:.1f}%), "
          f"acc gain={row['acc_gain']:.4f}, tree disagreement={row['disagreement']:.4f}")

# (d) final table
print("== Q3(d) table ==")
print(f"{'model':9s} {'clean acc':>9s} {'masked acc':>10s} {'bias2':>7s} {'variance':>8s}")
for m, s in main.items():
    print(f"{m:9s} {s['acc_mean']:9.4f} {s['masked_acc_mean']:10.4f} {s['bias2']:7.4f} {s['variance']:8.4f}")

dump("q3.json", dict(settings=settings, selected_models=main, depths=depth_rows))
