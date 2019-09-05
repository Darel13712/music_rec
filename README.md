# music_rec

This is a project for creating music recommender.

## Dataset
#### Features

I use an extraction from [Million Song Dataset](http://millionsongdataset.com/) 
or you can try [Million Song Subset](http://millionsongdataset.com/pages/getting-dataset/#subset) 
which is smaller. (Contains ≈ 37% of songs from train)

MSD provides following features (aggregated in `generate_data_from_msd.py`) 

An index file:

| song_id            | artist                   | title              |
| ------------------ | ------------------------ | ------------------ |
| SOPIFJT12A81C221AC | Bob Marley & The Wailers | Three Little Birds |

And a feature-file which contains audio features
that are 25 in total: 

- 12 timbre features 
- 12 chroma components
- max loudness value 

for every segment of the track.

So every song has a numpy vector associated to it of shape `(num_segmets, 25)`

#### User Preferences

I use user playcounts from [MSD Challenge](https://www.kaggle.com/c/msdchallenge/overview).

- [Train](http://millionsongdataset.com/tasteprofile/#getting) — Taste profile triples (user, song, count)
- [Test](http://millionsongdataset.com/challenge/#data1) — Visible and hidden parts from test files of EvalDataYear1. 
Visible is the query for user and hidden is the test. Corresponds to private part on Kaggle.

Competition used MAP@500, you can match yourself with the 
[leaderboard](https://www.kaggle.com/c/msdchallenge/leaderboard) if you use it.

Example:

| user                                     | song               | plays |
| ---------------------------------------- | ------------------ | ----- |
| 00007a02388c208ea7176479f6ae06f8224355b3 | SOAITVD12A6D4F824B | 3     |

### Getting data

MSD is very hard to handle because of large size (500GB) and lack of maintenance so today it is only available via 
[AWS snapshot](https://aws.amazon.com/ru/datasets/million-song-dataset/) which requires money to download.

I don't need all the data so I got only some of the features that I needed.
 
I also deleted songs that were [mismatched](http://millionsongdataset.com/blog/12-2-12-fixing-matching-errors/) 
so everything is nice and clean.

You can download my version of files here: 
- [Million Song Subset](https://yadi.sk/d/YF_rr29cZSVWJQ) - 1MB
- [Million Song Dataset](https://yadi.sk/d/1Rblxpppk69AMA) - 140MB
- [Ratings](https://yadi.sk/d/brk1sWOfqjUyZQ) - 490MB

May be someone will find this useful.