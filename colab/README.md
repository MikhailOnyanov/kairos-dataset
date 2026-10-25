# Colab pipeline

Three notebooks that run the whole solution on Google Colab, from the archive on Drive to the test predictions.

| notebook | runtime | what it does |
|---|---|---|
| `setup.ipynb` | CPU | copies the five core volumes into your Drive server side, checks SHA-256, unpacks texts, labels and the manifest, runs `kairos.label` and the baseline |
| `embed.ipynb` | T4 | embeds post text with multilingual-e5-base and post images with CLIP ViT-B/32 (images, document pages, article images, video posters), saves them to `MyDrive/kairos_work/` |
| `model.ipynb` | CPU | builds author relative features, runs the ablation on `GroupKFold` by author, trains the final model and writes the test predictions |

Run them in this order. The archive password is read from the Colab secret `KAIROS_PW`.

## Model

Every history feature reads only the author's earlier posts, as the level does.

1. Baseline features.
2. Author history: trend of the last one to three posts against the level, share of earlier posts that would be above or below, mean comments and reposts of earlier posts (never of the post itself).
3. Text and images: 32 PCA components each, similarity of the post to the author's earlier posts, and how the most similar earlier posts did (kNN over `log_reactions` minus the current level).
4. A gradient boosting classifier and a regressor on `resid`, blended as `0.5 z(regression) + 0.5 z(P_above − P_below)` with two cutoffs tuned on out of fold predictions.

Feature sets, blend weight and cutoffs were chosen on cross validation. Test was scored once per chosen model.

## Results

| | test macro-F1 | 95 % CI, resampling authors | QWK | accuracy |
|---|---:|---|---:|---:|
| baseline features, same environment | 0.5446 | | 0.4623 | 0.5508 |
| classifier with tuned class weights | 0.6494 | [0.639, 0.659] | 0.5792 | 0.6924 |
| **classifier and regressor blend** | **0.6551** | **[0.645, 0.665]** | **0.5915** | **0.6989** |

The blend beats the classifier on the same author resamples by [+0.001, +0.010].

Cross validation macro-F1 by feature set: baseline 0.5535, with author history 0.5891, with text 0.6298, with images 0.6328. The image gain is not significant (positive in four of five folds).

Predictions: `results/predictions/model_blend.csv`. Full numbers: `colab/results.json`.

```
python evaluate.py --predictions results/predictions/model_blend.csv
```
