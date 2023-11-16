
language_code = "en"
df_tmp = df.loc[df.lang == language_code, :].reset_index(drop=True)
pd.options.display.max_rows = 100  # see up to 100 entries
# widen how much text is presented of each tweet
pd.options.display.max_colwidth = 250
samples = df_tmp.sample(100)
samples