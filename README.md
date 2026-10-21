# ML Assignment 2: Bagging, Boosting and Bias Variance on Face Recognition

Roll number: M26MAC006 (seed = 26006)

## Files
- `common.py` - data split, masked test set, bagging and SAMME written from scratch
- `q1.py` - single tree baseline and bagging
- `q2.py` - SAMME AdaBoost
- `q3.py` - bias and variance over 30 resampled training sets
- `results/split.npz` - the saved train/val/test split
- `results/frozen.json` - frozen configurations 
- `results/.json`, `logs/.log` - final numbers and logs
- `figs/` - all figures

## Reproduce
```
pip install -r requirements.txt
python q1.py
python q2.py
python q3.py
```
or `bash run_all.sh`. Run them in order: q2 uses the models saved by q1, q3 uses the settings saved by q1 and q2.
Python 3.11, tested on Windows. q3 takes roughly an hour on 4 cores.
