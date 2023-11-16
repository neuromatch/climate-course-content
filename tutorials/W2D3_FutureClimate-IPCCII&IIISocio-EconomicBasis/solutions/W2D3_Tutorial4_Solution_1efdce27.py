
selected_words_2 = ["renewable", "wind", "solar", "geothermal", "biofuel"]

selectwords_detector_2 = re.compile(
    r"\b(?:{0})\b".format("|".join([str(word) for word in selected_words_2]))
)
df["select_talk_2"] = df.text.apply(
    lambda x: selectwords_detector_2.search(x, re.IGNORECASE)
)

selected_tweets_2 = df.loc[~df.select_talk_2.isnull(), :]
# selected_tweet_counts_2 = (
#     selected_tweets_2.created_at.groupby(
#         selected_tweets_2.created_at.apply(lambda x: x.date)
#     )
#     .count()
#     .rename("counts")
# )
# selected_tweet_fraction_2 = selected_tweet_counts_2 / total_tweetCounts
# selected_tweet_fraction_2 = ...

samples = selected_tweets_2.text.sample(100).values
samples