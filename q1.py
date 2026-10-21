import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

d = load_data()
Xtr, ytr, Xva, yva, Xte, yte, Xm = d["Xtr"], d["ytr"], d["Xva"], d["yva"], d["Xte"], d["yte"], d["Xtm"]
n = len(ytr)

# (a) single tree baseline
print("== Q1(a) single tree ==")
grid = []
for depth in [2, 4, 8, 16, None]:
    for leaf in [1, 2, 4]:
        params = dict(max_depth=depth, min_samples_leaf=leaf)
        tree = new_tree(params, SEED).fit(Xtr, ytr)
        row = dict(max_depth=depth, min_samples_leaf=leaf,
                   train=acc(tree.predict(Xtr), ytr), val=acc(tree.predict(Xva), yva),
                   test=acc(tree.predict(Xte), yte), **leaf_stats(tree, Xtr))
        grid.append(row)
        print(row)

best = max(grid, key=lambda r: (r["val"], -r["n_leaves"]))  # ties go to the smaller tree
tree_params = dict(max_depth=best["max_depth"], min_samples_leaf=best["min_samples_leaf"])
print("frozen tree:", best)
tree = new_tree(tree_params, SEED).fit(Xtr, ytr)
joblib.dump(tree, f"{MODELS}/frozen_tree.joblib")

fig, ax = plt.subplots(1, 2, figsize=(7, 2.6))
for leaf in [1, 2, 4]:
    rows = [r for r in grid if r["min_samples_leaf"] == leaf]
    depths = [str(r["max_depth"]) for r in rows]
    ax[0].plot(depths, [r["train"] for r in rows], "o-", label=f"train, leaf={leaf}")
    ax[0].plot(depths, [r["val"] for r in rows], "s--", label=f"val, leaf={leaf}")
    ax[1].plot(depths, [r["n_leaves"] for r in rows], "o-", label=f"leaf={leaf}")
ax[0].set_xlabel("max_depth")
ax[0].set_ylabel("accuracy")
ax[0].legend(fontsize=5)
ax[1].set_xlabel("max_depth")
ax[1].set_ylabel("number of leaves")
ax[1].legend(fontsize=6)
plt.tight_layout()
plt.savefig(f"{FIG}/q1a_depth.pdf")
plt.close()

# (b) bagging, (c) out of bag accuracy, disagreement and margin
print("== Q1(b, c) bagging ==")
Bs = [1, 5, 10, 25, 50, 100, 200]
bag = Bagging(200, tree_params, SEED + 1).fit(Xtr, ytr)  # B trees = first B trees of this pool
joblib.dump(bag, f"{MODELS}/bag_pool.joblib")
expected = n * (1 - (1 - 1 / n) ** n)
print(f"expected distinct examples per bootstrap: {expected:.2f}")

p_tr, p_va, p_te, p_m = bag.tree_preds(Xtr), bag.tree_preds(Xva), bag.tree_preds(Xte), bag.tree_preds(Xm)
bag_rows = []
for B in Bs:
    oob_votes = vote_counts(p_tr[:B], mask=~bag.inbag[:B])
    has_oob = oob_votes.sum(1) > 0
    row = dict(B=B,
               train=acc(vote_counts(p_tr[:B]), ytr), val=acc(vote_counts(p_va[:B]), yva),
               test=acc(vote_counts(p_te[:B]), yte), masked_test=acc(vote_counts(p_m[:B]), yte),
               mean_distinct=float(bag.n_distinct[:B].mean()), expected_distinct=expected,
               oob_acc=acc(oob_votes[has_oob], ytr[has_oob]), oob_covered=float(has_oob.mean()),
               disagreement=pairwise_disagreement(p_va[:B]),
               margin=float(margin(vote_counts(p_va[:B]) / B).mean()))
    bag_rows.append(row)
    print(row)

B_final = max(bag_rows, key=lambda r: r["val"])["B"]  
print("final B:", B_final)

fig, ax = plt.subplots(1, 3, figsize=(7, 2.3))
for a, key, title in zip(ax, ["oob_acc", "disagreement", "margin"],
                         ["OOB accuracy", "pairwise disagreement (val)", "mean vote margin (val)"]):
    a.plot(Bs, [r[key] for r in bag_rows], "o-")
    a.axvline(B_final, color="r", ls=":")
    a.set_xscale("log")
    a.set_xlabel("B")
    a.set_title(title, fontsize=7)
plt.tight_layout()
plt.savefig(f"{FIG}/q1c_oob.pdf")
plt.close()

# (d) frozen tree vs final bagged model
print("== Q1(d) tree vs bagging ==")
compare = {}
for name, X, imgs in [("clean", Xte, d["imgs"][d["te"]]), ("masked", Xm, d["imgs_te_masked"])]:
    tree_pred = tree.predict(X)
    tree_conf = tree.predict_proba(X).max(1)
    votes = bag.vote_frac(X, B_final)
    bag_pred = votes.argmax(1)
    bag_conf = votes.max(1)

    fixed = np.where((tree_pred != yte) & (bag_pred == yte))[0]
    both_wrong = np.where((tree_pred != yte) & (bag_pred != yte))[0]
    result = dict(tree_acc=acc(tree_pred, yte), bag_acc=acc(bag_pred, yte),
                  tree_conf=float(tree_conf.mean()), bag_conf=float(bag_conf.mean()),
                  n_fixed=len(fixed), n_both_wrong=len(both_wrong),
                  n_broken=int(((tree_pred == yte) & (bag_pred != yte)).sum()))
    print(name, result)

    cases = []
    picks = [("bag corrects tree", i) for i in fixed[:2]] + [("both wrong", i) for i in both_wrong[:2]]
    fig, ax = plt.subplots(1, len(picks), figsize=(7, 2), squeeze=False)
    for a, (kind, i) in zip(ax[0], picks):
        top3 = np.argsort(-votes[i])[:3]
        counts = {int(c): int(round(votes[i, c] * B_final)) for c in top3}
        case = dict(kind=kind, test_index=int(i), true=int(yte[i]),
                    tree_pred=int(tree_pred[i]), tree_conf=float(tree_conf[i]),
                    bag_pred=int(bag_pred[i]), bag_conf=float(bag_conf[i]),
                    top3_votes=counts, true_votes=int(round(votes[i, yte[i]] * B_final)))
        cases.append(case)
        print("  ", case)
        a.imshow(imgs[i], cmap="gray")
        a.axis("off")
        a.set_title(f"{kind}\ntrue {yte[i]}, tree {tree_pred[i]} ({tree_conf[i]:.2f})\n"
                    f"bag {bag_pred[i]}, votes {counts}", fontsize=4.5)
    plt.tight_layout()
    plt.savefig(f"{FIG}/q1d_{name}_cases.pdf")
    plt.close()
    result["cases"] = cases
    compare[name] = result

dump("q1.json", dict(grid=grid, frozen=best, bagging=bag_rows, B=B_final, comparison=compare))
dump("frozen.json", dict(tree_params=tree_params, B=B_final, seed=SEED))
