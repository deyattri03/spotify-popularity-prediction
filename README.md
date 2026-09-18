# spotify-popularity-prediction
# Predictive Classification under Severe Class Imbalance

Benchmarking four classifiers on 114,000 Spotify tracks where only 1 in 10 is popular.

## What this project is about

I wanted to know whether the audio features Spotify publishes for every track (danceability, energy, loudness, valence, tempo and so on) can tell you if a song will be popular.
They mostly can't. That turned out to be the more interesting result, and most of the work here went into making sure I could say that with confidence rather than just getting a high-looking number.
Two things about this dataset trip people up. About a fifth of the rows are duplicates, and if you don't remove them the same song ends up in both your training and test sets, which quietly inflates every score you report. And only 9.9% of tracks are popular, so accuracy is close to useless: predict "not popular" for everything and you score 90.1% while finding nothing at all.

## The data

I used the [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) from Kaggle. It has 114,000 tracks, 21 columns and 114 genres. Each row gives the audio features plus artist, album, genre and an explicit flag.
A track counts as popular if its Spotify popularity score is above 60 on the 0 to 100 scale.

## Cleaning

| Step | Rows left | Removed |
|---|---:|---:|
| Starting dataset | 114,000 | |
| Dropped rows with missing track, artist or album | 113,999 | 1 |
| Removed duplicate tracks | 89,740 | 24,259 |
| Removed impossible values | 89,405 | 335 |

**Duplicates.** The dataset is built as 114 genres with 1,000 tracks each, so the same song shows up several times under different genre labels. That's 24,259 rows, about 21% of the data. If you leave them in, near-identical rows land on both sides of the train/test split and your scores go up for no real reason.

**Impossible values.** There's a group of rows with tempo exactly 0 and loudness around −50 dB. No real recording looks like that. They show up as a straight horizontal line if you plot loudness against tempo. I also dropped anything under 30 seconds or over 15 minutes.

## Features

I ended up with 37 features from the 9 raw audio columns plus the metadata that usually gets thrown away.

**Genre** is the big one. I encoded it by target encoding with smoothing, fitted only on the training fold so nothing leaks from the test set. Rare genres get pulled toward the overall average so they don't produce wild encodings off three or four songs.

**Columns most people discard:** `explicit`, `key` (one-hot), `mode`, `time_signature`. They're sitting right there in the raw data.

**Transforms:** log1p on `speechiness`, `liveness` and `instrumentalness`, all of which are heavily skewed.

**Interactions that made musical sense:** valence × energy (call it mood), energy × |loudness| (intensity), danceability × energy, and acousticness relative to energy.

**Flags:** whether a track is instrumental, whether it's live, which tempo band it falls in, and how many tracks the artist has in the dataset.

Everything is fitted after the split. Nothing sees the test set before it should.

## Models and how I judged them

Four classifiers on an 80/20 stratified split: logistic regression as a baseline, k-nearest neighbours, random forest, and XGBoost. Imbalance handled with class weights, and `scale_pos_weight` for XGBoost.

**I ranked on PR-AUC, not accuracy.** With 9.9% positives, the do-nothing baseline is already 90.1% accurate. Any accuracy number has to be read against that, not against 50%. PR-AUC has its own baseline of 0.099, so you can see straight away whether a model is doing anything.

### Results

XGBoost came out on top on every measure that accounts for the imbalance.

| | XGBoost | Baseline |
|---|---:|---:|
| PR-AUC | **0.466** | 0.099 |
| ROC-AUC | **0.879** | 0.500 |
| Recall on popular tracks | **0.76** | 0.00 |
| Accuracy | — | 0.901 |

Five-fold stratified cross-validation put XGBoost at **PR-AUC 0.478 ± 0.006**, so the result is stable across folds rather than an artefact of one split.
Tuning the decision threshold on F1 instead of leaving it at 0.50 pushed F1 on the popular class from **0.447 to 0.466**, with the best cut at 0.63.
The full comparison across all four models, with confusion matrices, per-class precision and recall, ROC and precision-recall curves, and feature importances, is in the notebook.

## What I found

**Accuracy picks the wrong model here.** The most accurate model in the comparison scores about 0.905, which is a fraction of a point above doing nothing at all, and it finds roughly a quarter of the popular tracks. XGBoost, chosen on PR-AUC, finds 76% of them. If I had ranked on accuracy I would have picked the weaker model and reported a number that sounded fine.

**Genre matters far more than how a song sounds.** Encoded genre accounts for roughly 24% of XGBoost's total gain and 36% of random forest importance. The individual audio features each contribute very little. What category a track sits in tells you more than its measurable sound does.

**The duplicates were the biggest problem.** An earlier version of this analysis, before I deduplicated and while I was still reporting accuracy, gave 93.97%. That number was wrong twice over: inflated by leakage, and meaningless against a 90.1% baseline.

**t-SNE and PCA agree.** Project the features down to two dimensions and popular tracks sit right on top of everything else. No separate cluster, no visible structure. That lines up with the modest supervised results.

## What this doesn't tell you

Spotify's popularity score moves over time and is partly driven by their own algorithms, so it's a snapshot of the platform rather than a fixed property of a song.
The cutoff at 60 is arbitrary. Predicting the raw 0 to 100 score and measuring rank correlation would keep information that turning it into a yes/no throws away.
Target encoding still carries some leakage risk for genres with few tracks. Smoothing reduces it, doesn't remove it.
There's no release date in the data, so I couldn't split chronologically or check whether the models hold up over time.
One thing worth flagging: `is_live` comes out high in XGBoost's importance but near zero in the random forest's. That usually means the boosting model is using it in a handful of splits rather than it being genuinely predictive, so I wouldn't read anything into it.

---

**Built with:** Python, pandas, NumPy, scikit-learn, XGBoost, Matplotlib, Seaborn
