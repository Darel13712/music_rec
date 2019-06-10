# music_rec

This is a project for creating music recommender.

## Dataset
#### Features

The plan is to use [Million Song Dataset](http://millionsongdataset.com/) 
but at the moment I'm working with 
[Million Song Subset](http://millionsongdataset.com/pages/getting-dataset/#subset) 
which is easier to download and smaller. (Contains ≈ 37% of songs from train)

MSD provides features (aggregated in `generate_data_from_msd.py`) 
that are 25 in total: 12 timbre features, 
12 chroma components and a max loudness value 
for every segment of the track.

#### User Preferences

I use user playcounts from [MSD Challenge](https://www.kaggle.com/c/msdchallenge/overview).

- [Train](http://millionsongdataset.com/tasteprofile/#getting) — Taste profile triples (user, song, count)
- [Test](http://millionsongdataset.com/challenge/#data1) — Visible and hidden parts from test files of EvalDataYear1. 
Visible is the query for user and hidden is the test. Corresponds to private part on Kaggle.

Competition used MAP@500, you can match yourself with the 
[leaderboard](https://www.kaggle.com/c/msdchallenge/leaderboard) if you use it.

### Getting data

You can download my clean files [here](https://yadi.sk/d/aaJAoFTjao79sQ)
