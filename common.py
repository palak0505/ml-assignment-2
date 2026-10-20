import os
import json
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.tree import DecisionTreeClassifier

ROLL = "M26MAC006"
SEED = int("".join(ch for ch in ROLL if ch.isdigit()))  # 26006
K = 40

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figs")
MODELS = os.path.join(HERE, "models")
for folder in (RES, FIG, MODELS):
    os.makedirs(folder, exist_ok=True)


def load_data():
    data = fetch_olivetti_faces()
    imgs, y = data.images.astype(np.float64), data.target
    split_file = os.path.join(RES, "split.npz")
    if os.path.exists(split_file):
        s = np.load(split_file)
        tr, va, te = s["train"], s["val"], s["test"]
    else:
        rng = np.random.default_rng(SEED)
        tr, va, te = [], [], []
        for c in range(K):
            idx = rng.permutation(np.where(y == c)[0])
            tr += list(idx[:6])
            va += list(idx[6:8])
            te += list(idx[8:])
        tr, va, te = np.array(tr), np.array(va), np.array(te)
        np.savez(split_file, train=tr, val=va, test=te, seed=SEED)

    masked = imgs[te].copy()
    masked[:, 24:40, 24:40] = 0.0
    flat = lambda a: a.reshape(len(a), -1)
    return dict(imgs=imgs, y=y, tr=tr, va=va, te=te,
                Xtr=flat(imgs[tr]), ytr=y[tr], Xva=flat(imgs[va]), yva=y[va],
                Xte=flat(imgs[te]), yte=y[te], Xtm=flat(masked), imgs_te_masked=masked)


def new_tree(params, seed):
    return DecisionTreeClassifier(random_state=int(seed), **params)


def leaf_stats(tree, X):
    counts = np.bincount(tree.apply(X))
    counts = counts[counts > 0]
    leaves = tree.get_n_leaves()
    return dict(n_leaves=int(leaves), mean_samples_per_leaf=len(X) / leaves,
                frac_singleton_leaves=float((counts == 1).sum() / leaves), depth=int(tree.get_depth()))


def vote_counts(preds, k=K, mask=None):
    # preds has shape (trees, samples); returns (samples, k)
    votes = np.zeros((preds.shape[1], k))
    for c in range(k):
        hit = preds == c
        if mask is not None:
            hit &= mask
        votes[:, c] = hit.sum(0)
    return votes


def margin(frac):
    s = np.sort(frac, axis=1)
    return s[:, -1] - s[:, -2]


def pairwise_disagreement(preds, per_sample=False):
    T = preds.shape[0]
    if T < 2:
        return np.zeros(preds.shape[1]) if per_sample else 0.0
    c = vote_counts(preds)
    d = 1 - (c * (c - 1)).sum(1) / (T * (T - 1))
    return d if per_sample else float(d.mean())


def acc(scores, y):
    scores = np.asarray(scores)
    pred = scores.argmax(1) if scores.ndim == 2 else scores
    return float((pred == y).mean())


def dump(name, obj):
    with open(os.path.join(RES, name), "w") as f:
        json.dump(obj, f, indent=2, default=float)


class Bagging:
    def __init__(self, B, params, seed):
        self.B, self.params, self.seed = B, params, seed

    def fit(self, X, y):
        n = len(y)
        rng = np.random.default_rng(self.seed)
        self.boot = rng.integers(0, n, size=(self.B, n))
        seeds = rng.integers(0, 2**31 - 1, size=self.B)
        self.trees = [new_tree(self.params, s).fit(X[idx], y[idx]) for idx, s in zip(self.boot, seeds)]
        self.inbag = np.zeros((self.B, n), bool)
        for b in range(self.B):
            self.inbag[b, self.boot[b]] = True
        self.n_distinct = np.array([len(np.unique(idx)) for idx in self.boot])
        return self

    def tree_preds(self, X, T=None):
        T = T or len(self.trees)
        return np.stack([t.predict(X) for t in self.trees[:T]])

    def vote_frac(self, X, T=None):
        preds = self.tree_preds(X, T)
        return vote_counts(preds) / len(preds)


class SAMME:
    def __init__(self, depth, rounds, seed, k=K):
        self.depth, self.rounds, self.seed, self.k = depth, rounds, seed, k

    def fit(self, X, y):
        n = len(y)
        rng = np.random.default_rng(self.seed)
        w = np.ones(n) / n
        self.trees, self.alphas, self.errors, self.weights, self.wrong = [], [], [], [], []
        for t in range(self.rounds):
            tree = DecisionTreeClassifier(max_depth=self.depth, random_state=int(rng.integers(2**31 - 1)))
            tree.fit(X, y, sample_weight=w)
            wrong = tree.predict(X) != y
            err = w[wrong].sum()
            if err >= 1 - 1 / self.k:
                # not better than random guessing, so alpha would be <= 0: drop it and stop
                print(f"round {t + 1}: error {err:.3f} >= 1 - 1/K, stopping")
                break
            alpha = np.log((1 - err) / max(err, 1e-10)) + np.log(self.k - 1)
            self.trees.append(tree)
            self.alphas.append(alpha)
            self.errors.append(err)
            self.weights.append(w)
            self.wrong.append(wrong)
            w = w * np.exp(alpha * wrong)
            w = w / w.sum()
            if err == 0:
                break
        self.final_w = w
        self.alphas = np.array(self.alphas)
        self.errors = np.array(self.errors)
        self.weights = np.array(self.weights)
        self.wrong = np.array(self.wrong)
        return self

    def learner_preds(self, X, T=None):
        T = T or len(self.trees)
        return np.stack([t.predict(X) for t in self.trees[:T]])

    def vote(self, X, T=None):
        T = T or len(self.trees)
        S = np.zeros((len(X), self.k))
        for tree, a in zip(self.trees[:T], self.alphas[:T]):
            S[np.arange(len(X)), tree.predict(X)] += a
        return S / self.alphas[:T].sum()
