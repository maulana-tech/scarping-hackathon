
import pandas as pd
import numpy as np
import os
import re
from sklearn.feature_extraction.text import CountVectorizer

# Define paths
DATA_PATH = '../data/'
DATA_ACTUAL_PATH = '../data/processed/'

print("Loading datasets...")
try:
    df_youtube = pd.read_csv(os.path.join(DATA_PATH, 'Scraper Youtube CoreTax.csv'))
    print(f"YouTube Data: {len(df_youtube)} rows")
except Exception as e:
    print(f"Error loading YouTube data: {e}")
    df_youtube = pd.DataFrame()

try:
    df_playstore = pd.read_csv(os.path.join(DATA_PATH, 'CoreTax Scraper M Pajak 2025.csv'))
    print(f"Play Store Data: {len(df_playstore)} rows")
except Exception as e:
    print(f"Error loading Play Store data: {e}")
    df_playstore = pd.DataFrame()

try:
    df_combined_old = pd.read_csv(os.path.join(DATA_ACTUAL_PATH, 'CoreTax Combined Data Clean.csv'))
    print(f"Existing Combined Data: {len(df_combined_old)} rows")
except Exception as e:
    print(f"Error loading Combined data: {e}")
    df_combined_old = pd.DataFrame()

# Standardize Columns
if not df_youtube.empty:
    df_youtube = df_youtube[['date', 'text', 'source']].copy()
    df_youtube['rating'] = np.nan

if not df_playstore.empty:
    df_playstore = df_playstore.rename(columns={'at': 'date', 'content': 'text'})
    df_playstore = df_playstore[['date', 'text', 'source', 'rating']].copy()

if not df_combined_old.empty:
    if 'date' not in df_combined_old.columns:
        df_combined_old['date'] = np.nan
    if 'rating' not in df_combined_old.columns:
        df_combined_old['rating'] = np.nan
    df_combined_old = df_combined_old[['date', 'text', 'source', 'rating']].copy()

# Combine
df_combined = pd.concat([df_youtube, df_playstore, df_combined_old], ignore_index=True)
df_combined = df_combined.drop_duplicates(subset=['text'])

print(f"\nTotal Combined Data (Unique Texts): {len(df_combined)}")
print("\nSource Distribution:")
print(df_combined['source'].value_counts())

# Basic Text Analysis
def clean_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = text.lower().strip()
    return text

df_combined['cleaned_text'] = df_combined['text'].apply(clean_text)

# Top Bigrams
vec = CountVectorizer(ngram_range=(2, 2), stop_words=None).fit(df_combined['cleaned_text'])
bag_of_words = vec.transform(df_combined['cleaned_text'])
sum_words = bag_of_words.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

print("\nTop 10 Bigrams:")
for word, freq in words_freq[:10]:
    print(f"{word}: {freq}")
