import pandas as pd
import os
from . import config

def load_and_merge_data():
    """
    Loads data from Play Store, YouTube, and Social Media (Twitter/TikTok),
    renames columns, and merges them into a single DataFrame.
    """
    print("Loading datasets...")
    
    # Load Dataframes
    # Handle missing files gracefully
    if os.path.exists(config.FILE_PLAYSTORE):
        df1 = pd.read_csv(config.FILE_PLAYSTORE)
    else:
        print(f"Warning: {config.FILE_PLAYSTORE} not found. Creating empty DataFrame.")
        df1 = pd.DataFrame(columns=['source', 'content'])

    if os.path.exists(config.FILE_YOUTUBE):
        df2 = pd.read_csv(config.FILE_YOUTUBE)
    else:
        print(f"Warning: {config.FILE_YOUTUBE} not found. Creating empty DataFrame.")
        df2 = pd.DataFrame(columns=['source', 'text'])

    if os.path.exists(config.FILE_TWITTER_TIKTOK):
        df3 = pd.read_csv(config.FILE_TWITTER_TIKTOK)
    else:
        print(f"Warning: {config.FILE_TWITTER_TIKTOK} not found. Creating empty DataFrame.")
        df3 = pd.DataFrame(columns=['source', 'cleaned_text'])

    # Rename columns to match 'content'
    
    # df1 (Play Store): has 'content', 'at' (date), 'rating', 'source'
    # No rename needed for content, but let's be safe
    if 'text' in df1.columns:
        df1.rename(columns={'text': 'content'}, inplace=True)

    # df2 (YouTube): has 'text', 'date', 'source'
    if 'text' in df2.columns:
        df2.rename(columns={'text': 'content'}, inplace=True)
    
    # df3 (Twitter/TikTok): has 'cleaned_text', 'sentiment', 'sentiment_score', 'source'
    if 'cleaned_text' in df3.columns:
        df3.rename(columns={'cleaned_text': 'content'}, inplace=True)
    elif 'text' in df3.columns:
         df3.rename(columns={'text': 'content'}, inplace=True)

    # Merge
    print("Merging datasets...")
    df = pd.concat([df1, df2, df3], axis=0, ignore_index=True)
    
    print(f"Data Info after merge:")
    print(df.info())
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Duplicates: {df.duplicated().sum()}")
    
    # Drop unnecessary columns if they exist
    cols_to_drop = ['rating', 'at', 'date', 'sentiment', 'sentiment_score']
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    
    return df
