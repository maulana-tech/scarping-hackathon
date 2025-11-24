# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#     "pandas",
#     "torch",
#     "transformers",
#     "scikit-learn",
#     "emoji",
#     "tqdm",
#     "numpy<2",
#     "protobuf",
# ]
# ///

import pandas as pd
import re
import emoji
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import torch
import os

# Check if CUDA/MPS is available for faster processing
device = 0 if torch.cuda.is_available() else -1
if torch.backends.mps.is_available():
    try:
        # MPS is supported in newer transformers/torch versions
        device = "mps"
    except:
        device = -1

print(f"Using device: {device}")

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user mentions (@username)
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags (#hashtag) - remove symbol only
    text = re.sub(r'#', '', text)
    
    # Remove special characters and numbers (keep letters and basic punctuation)
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove emojis
    text = emoji.replace_emoji(text, replace='')
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def load_data():
    data_frames = []
    
    # Load Twitter Data
    if os.path.exists('data/CoreTax-Twitter-01.csv'):
        try:
            df_twitter = pd.read_csv('data/CoreTax-Twitter-01.csv')
            if 'full_text' in df_twitter.columns:
                temp_df = df_twitter[['full_text', 'created_at']].rename(columns={'full_text': 'text', 'created_at': 'date'})
                temp_df['source'] = 'Twitter'
                data_frames.append(temp_df)
        except Exception as e:
            print(f"Error loading Twitter data: {e}")

    # Load TikTok Comments
    if os.path.exists('data/Tiktok-comment.csv'):
        try:
            df_tiktok_comment = pd.read_csv('data/Tiktok-comment.csv')
            if 'text' in df_tiktok_comment.columns:
                temp_df = df_tiktok_comment[['text', 'createTimeISO']].rename(columns={'createTimeISO': 'date'})
                temp_df['source'] = 'TikTok Comment'
                data_frames.append(temp_df)
        except Exception as e:
            print(f"Error loading TikTok comments: {e}")

    # Load TikTok Video Captions
    if os.path.exists('data/TiktokVideo-01.csv'):
        try:
            df_tiktok_video = pd.read_csv('data/TiktokVideo-01.csv')
            if 'text' in df_tiktok_video.columns:
                temp_df = df_tiktok_video[['text', 'createTimeISO']].rename(columns={'createTimeISO': 'date'})
                temp_df['source'] = 'TikTok Video'
                data_frames.append(temp_df)
        except Exception as e:
            print(f"Error loading TikTok video data: {e}")
    
    if not data_frames:
        raise ValueError("No data loaded! Check if files exist in data/ directory.")
        
    full_df = pd.concat(data_frames, ignore_index=True)
    return full_df

def analyze_sentiment(df):
    print("Loading IndoBERT model...")
    # Using a popular pre-trained sentiment model for Indonesian
    model_name = "ayameRushia/bert-base-indonesian-1.5G-sentiment-analysis-smsa"
    
    # Pipeline handles device automatically if passed correctly
    sentiment_pipeline = pipeline("sentiment-analysis", model=model_name, device=device)
    
    print("Predicting sentiment...")
    results = []
    texts = df['cleaned_text'].tolist()
    
    # Process in batches
    batch_size = 32
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        # Handle empty strings or very short strings by replacing them temporarily or skipping
        # The pipeline might fail on empty strings
        processed_batch = [t if len(t.strip()) > 0 else "." for t in batch]
        
        try:
            predictions = sentiment_pipeline(processed_batch, truncation=True, max_length=512)
            results.extend(predictions)
        except Exception as e:
            print(f"Error in batch {i}: {e}")
            # Fallback
            results.extend([{'label': 'neutral', 'score': 0.0}] * len(batch))
            
    df['sentiment'] = [r['label'] for r in results]
    df['score'] = [r['score'] for r in results]
    
    return df

def extract_keywords(df):
    print("Extracting keywords using TF-IDF...")
    keywords_by_sentiment = {}
    
    sentiments = df['sentiment'].unique()
    
    stop_words_indo = [
        'dan', 'di', 'ke', 'dari', 'yang', 'ini', 'itu', 'untuk', 'pada', 'adalah', 
        'saya', 'aku', 'kamu', 'dia', 'kita', 'mereka', 'apa', 'siapa', 'kapan', 
        'dimana', 'kenapa', 'bagaimana', 'bisa', 'ada', 'jadi', 'akan', 'sudah', 
        'belum', 'tidak', 'bukan', 'tapi', 'tetapi', 'karena', 'jika', 'kalau',
        'nanti', 'saja', 'juga', 'dengan', 'atau', 'ya', 'yuk', 'yg', 'gak', 'gk',
        'dgn', 'sdh', 'blm', 'kalo', 'klo', 'tp', 'jd', 'nih', 'tuh', 'dong', 'sih',
        'kok', 'kan', 'deh', 'lah', 'pun', 'mah', 'biar', 'pas', 'lagi', 'bgt',
        'banget', 'aja', 'doang', 'cuma', 'hanya', 'sama', 'banyak', 'sedikit',
        'lebih', 'kurang', 'paling', 'sangat', 'terlalu', 'cukup', 'perlu', 'harus',
        'mau', 'ingin', 'hendak', 'bakal', 'boleh', 'dapat', 'mungkin', 'tentu',
        'pasti', 'yakin', 'tahu', 'tau', 'mengerti', 'paham', 'lihat', 'dengar',
        'baca', 'tulis', 'bicara', 'kata', 'bilang', 'ucap', 'sebut', 'tanya',
        'jawab', 'minta', 'beri', 'kasih', 'ambil', 'bawa', 'taruh', 'simpan',
        'buang', 'hapus', 'ubah', 'ganti', 'tambah', 'kurang', 'bagi', 'kali',
        'coretax', 'pajak', 'djp', 'kpp', 'kantor', 'pelayanan', 'admin', 'min',
        'nya', 'sy', 'gw', 'gue', 'lu', 'lo', 'ga', 'tak', 'mas', 'mbak', 'kak', 'pak', 'bu'
    ]
    
    for sentiment in sentiments:
        subset = df[df['sentiment'] == sentiment]
        text_data = subset['cleaned_text'].dropna().tolist()
        # Filter out empty strings
        text_data = [t for t in text_data if len(t.strip()) > 0]
        
        if not text_data:
            continue
            
        try:
            # Use a smaller max_features to focus on top words
            vectorizer = TfidfVectorizer(max_features=100, stop_words=stop_words_indo)
            tfidf_matrix = vectorizer.fit_transform(text_data)
            feature_names = vectorizer.get_feature_names_out()
            
            sums = tfidf_matrix.sum(axis=0)
            data = []
            for col, term in enumerate(feature_names):
                data.append( (term, sums[0, col]) )
            
            ranking = sorted(data, key=lambda x: x[1], reverse=True)
            keywords_by_sentiment[sentiment] = [term for term, score in ranking[:15]]
        except ValueError:
            keywords_by_sentiment[sentiment] = []
            
    return keywords_by_sentiment

def main():
    # 1. Load Data
    print("Loading data...")
    try:
        df = load_data()
    except ValueError as e:
        print(e)
        return
        
    print(f"Total records loaded: {len(df)}")
    
    # 2. Preprocessing
    print("Cleaning text...")
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Remove empty rows after cleaning
    df = df[df['cleaned_text'].str.len() > 2]
    print(f"Records after cleaning: {len(df)}")
    
    # 3. Sentiment Analysis (IndoBERT)
    df = analyze_sentiment(df)
    
    # 4. Keyword Extraction (TF-IDF)
    keywords = extract_keywords(df)
    
    # 5. Output
    output_path = 'data/sentiment_results.csv'
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    
    print("\n=== Summary ===")
    print(df['sentiment'].value_counts())
    
    print("\n=== Top Keywords by Sentiment ===")
    for sentiment, keys in keywords.items():
        print(f"\n{sentiment.upper()}:")
        print(", ".join(keys))

if __name__ == "__main__":
    main()
