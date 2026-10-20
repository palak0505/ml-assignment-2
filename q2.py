import json
import time
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

np.set_printoptions(precision=4, suppress=True)
d = load_data()
Xtr, ytr, Xva, yva, Xte, yte, Xm = d["Xtr"], d["ytr"], d["Xva"], d["yva"], d["Xte"], d["yte"], d["Xtm"]
frozen = json.load(open(f"{RES}/frozen.json"))
tree_params, B = frozen["tree_params"], frozen["B"]

# (a) one SAMME round on a small 3-class example
print("== Q2(a) toy example ==")
Xt = np.array([[1, 1], [1.5, 2], [2, 1.2], [5, 5], [5.5, 4.5], [6, 5.2], [1, 6], [3.2, 5.5]])
yt = np.array([0, 0, 0, 1, 1, 1, 2, 2])
toy = SAMME(depth=1, rounds=1, seed=SEED, k=3).fit(Xt, yt)
print("labels          :", yt)
print("initial weights :", toy.weights[0])
print("predictions     :", toy.trees[0].predict(Xt))
print("misclassified   :", toy.wrong[0].astype(int))
print(f"weighted error  : {toy.errors[0]:.4f}")
print(f"alpha           : log((1-e)/e) + log(K-1) = {toy.alphas[0]:.4f}")
print("updated weights :", toy.final_w)

# (b) choose depth and rounds on clean validation accuracy
print("== Q2(b) depth / rounds ==")
grid = []
for depth in [1, 2, 3, 4]:
    m = SAMME(depth, 100, SEED).fit(Xtr, ytr)
    for T in [10, 25, 50, 100]:
        row = dict(depth=depth, rounds=T, train=acc(m.vote(Xtr, T), ytr), val=acc(m.vote(Xva, T), yva))
        grid.append(row)
        print(row)
best = max(grid, key=lambda r: r["val"])  # ties go to the smaller depth / fewer rounds
bd, bT = best["depth"], best["rounds"]
print("selected:", best)

boost = SAMME(bd, bT, SEED).fit(Xtr, ytr)
joblib.dump(boost, f"{MODELS}/boost_selected.joblib")

rounds = np.arange(1, len(boost.alphas) + 1)
top10 = np.sort(boost.weights, axis=1)[:, -10:].sum(1)
fig, ax = plt.subplots(1, 4, figsize=(7.2, 2))
ax[0].plot(rounds, boost.errors); ax[0].set_title("weighted training error", fontsize=7)
ax[1].plot(rounds, boost.alphas); ax[1].set_title("learner weight alpha", fontsize=7)
ax[2].plot(rounds, boost.weights.max(1)); ax[2].set_title("max sample weight", fontsize=7)
ax[3].plot(rounds, top10); ax[3].set_title("weight of 10 heaviest images", fontsize=7)
for a in ax:
    a.set_xlabel("round", fontsize=7)
    a.tick_params(labelsize=6)
plt.tight_layout()
plt.savefig(f"{FIG}/q2b_rounds.pdf")
plt.close()

# (c) boosting vs bagging with the same number of learners
print("== Q2(c) boosting vs bagging ==")
compare = []
for T in [25, 50, 100]:
    start = time.time()
    bag = Bagging(T, tree_params, SEED + 1).fit(Xtr, ytr)
    bag_time = time.time() - start
    start = time.time()
    bst = SAMME(bd, T, SEED).fit(Xtr, ytr)
    boost_time = time.time() - start
    row = dict(learners=T,
               bag_train=acc(bag.vote_frac(Xtr), ytr), bag_val=acc(bag.vote_frac(Xva), yva),
               bag_test=acc(bag.vote_frac(Xte), yte), bag_time=bag_time,
               boost_train=acc(bst.vote(Xtr), ytr), boost_val=acc(bst.vote(Xva), yva),
               boost_test=acc(bst.vote(Xte), yte), boost_time=boost_time)
    compare.append(row)
    print(row)

# training face with the largest weight and its prediction history
heavy = int(boost.weights.max(0).argmax())
w_hist = boost.weights[:, heavy]
wrong_hist = boost.wrong[:, heavy]
heavy_info = dict(train_index=heavy, identity=int(ytr[heavy]), max_weight=float(w_hist.max()),
                  round_of_max=int(w_hist.argmax() + 1), uniform_weight=1 / len(ytr),
                  rounds_wrong=int(wrong_hist.sum()), rounds=len(wrong_hist),
                  avg_wrong_rate_all_faces=float(boost.wrong.mean()),
                  ensemble_correct=bool(boost.vote(Xtr[heavy:heavy + 1]).argmax() == ytr[heavy]))
print("heaviest face:", heavy_info)
print("weak learner wrong on it (1 = wrong):", wrong_hist.astype(int))

fig, ax = plt.subplots(1, 3, figsize=(7.2, 2), gridspec_kw=dict(width_ratios=[1, 3, 3]))
ax[0].imshow(d["imgs"][d["tr"][heavy]], cmap="gray")
ax[0].axis("off")
ax[0].set_title(f"identity {ytr[heavy]}", fontsize=7)
ax[1].semilogy(rounds, w_hist)
ax[1].set_title("weight of this face", fontsize=7)
ax[2].scatter(rounds, wrong_hist.astype(int), s=6)
ax[2].set_yticks([0, 1])
ax[2].set_yticklabels(["right", "wrong"], fontsize=6)
ax[2].set_title("weak learner on this face", fontsize=7)
for a in ax[1:]:
    a.set_xlabel("round", fontsize=7)
    a.tick_params(labelsize=6)
plt.tight_layout()
plt.savefig(f"{FIG}/q2c_heavy.pdf")
plt.close()

# (d) frozen models on clean and masked test sets
print("== Q2(d) clean vs masked ==")
tree = joblib.load(f"{MODELS}/frozen_tree.joblib")
bag = joblib.load(f"{MODELS}/bag_pool.joblib")
scores = {}
final = {}
for name, X in [("clean", Xte), ("masked", Xm)]:
    scores[name] = dict(tree=tree.predict_proba(X), bag=bag.vote_frac(X, B), boost=boost.vote(X))
    final[name] = {m: dict(acc=acc(s, yte), mean_winning_vote=float(s.max(1).mean()))
                   for m, s in scores[name].items()}
    print(name, final[name])

# face where masking changes one model's answer but not the other's
ok = {m: {n: scores[n][m].argmax(1) == yte for n in scores} for m in ("bag", "boost")}
bag_flips = ok["bag"]["clean"] != ok["bag"]["masked"]
boost_flips = ok["boost"]["clean"] != ok["boost"]["masked"]
both_right_clean = ok["bag"]["clean"] & ok["boost"]["clean"]
cands = np.where((bag_flips != boost_flips) & both_right_clean)[0]
if len(cands) == 0:
    cands = np.where(bag_flips != boost_flips)[0]
i = int(cands[0])
c = yte[i]

bag_clean, bag_masked = bag.tree_preds(Xte[i:i + 1], B)[:, 0], bag.tree_preds(Xm[i:i + 1], B)[:, 0]
bst_clean, bst_masked = boost.learner_preds(Xte[i:i + 1])[:, 0], boost.learner_preds(Xm[i:i + 1])[:, 0]
avg_w = boost.weights.mean(0)
face = dict(test_index=i, identity=int(c),
            bag_pred=[int(scores["clean"]["bag"][i].argmax()), int(scores["masked"]["bag"][i].argmax())],
            boost_pred=[int(scores["clean"]["boost"][i].argmax()), int(scores["masked"]["boost"][i].argmax())],
            bag_true_vote=[float(scores["clean"]["bag"][i, c]), float(scores["masked"]["bag"][i, c])],
            boost_true_vote=[float(scores["clean"]["boost"][i, c]), float(scores["masked"]["boost"][i, c])],
            bag_disagreement=[float(pairwise_disagreement(bag_clean[:, None])),
                              float(pairwise_disagreement(bag_masked[:, None]))],
            bag_trees_changed=float((bag_clean != bag_masked).mean()),
            boost_learners_changed=float((bst_clean != bst_masked).mean()),
            boost_alpha_changed=float(boost.alphas[bst_clean != bst_masked].sum() / boost.alphas.sum()),
            boost_weight_on_identity=float(avg_w[ytr == c].sum()), uniform_identity_weight=6 / len(ytr))
print("face (values are [clean, masked]):", face)

fig, ax = plt.subplots(1, 2, figsize=(3.4, 2))
ax[0].imshow(d["imgs"][d["te"][i]], cmap="gray")
ax[0].set_title(f"clean: bag {face['bag_pred'][0]}, boost {face['boost_pred'][0]}", fontsize=6)
ax[1].imshow(d["imgs_te_masked"][i], cmap="gray")
ax[1].set_title(f"masked: bag {face['bag_pred'][1]}, boost {face['boost_pred'][1]}", fontsize=6)
for a in ax:
    a.axis("off")
plt.tight_layout()
plt.savefig(f"{FIG}/q2d_face.pdf")
plt.close()

dump("q2.json", dict(grid=grid, selected=best, compare=compare, heaviest=heavy_info, final=final, face=face))
frozen.update(boost_depth=bd, boost_rounds=bT)
dump("frozen.json", frozen)
