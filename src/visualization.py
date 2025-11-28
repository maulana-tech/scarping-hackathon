import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import os
from . import config

def plot_sentiment_distribution(df):
    sentiment_count = df['sentiment'].value_counts()
    sns.set_style('whitegrid')
    
    fig, ax = plt.subplots(figsize=(8,6))
    ax = sns.barplot(x=sentiment_count.index, y=sentiment_count.values, palette='viridis')
    
    plt.title('Sentimen CoreTax', fontsize=14)
    total = len(df['sentiment'])
    
    for i, count in enumerate(sentiment_count.values):
        percentage = f'{100 * count / total:.2f}%'
        ax.text(i, count + 0.10, f'{count}\n({percentage})', ha='center', va='bottom', fontsize=8)
    
    plt.savefig(os.path.join(config.OUTPUTS_DIR, 'sentiment_distribution.png'))
    # plt.show() # Commented out for script execution

def plot_sentiment_by_source(df):
    plt.figure(figsize=(10,6))
    ax = sns.countplot(data=df, x="source", hue="sentiment", palette="viridis")
    group_totals = df.groupby("source")["sentiment"].count()
    
    for p in ax.patches:
        height = p.get_height()
        if height == 0: continue
        
        x = p.get_x() + p.get_width() / 2
        # Handle potential index error if source not found in labels
        try:
            source = ax.get_xticklabels()[int(x)].get_text()
            total = group_totals[source]
            percentage = 100 * height / total
            ax.annotate(f'{percentage:.1f}%', (x, height), ha='center', va='bottom', fontsize=9, color='black')
        except:
            pass

    plt.title("Distribusi Sentiment Setiap Sumber")
    plt.xlabel("Source")
    plt.ylabel(" ")
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUTS_DIR, 'sentiment_by_source.png'))

def generate_wordclouds(df):
    positive_texts = ' '.join(df[df['sentiment'] == 'positive']['stemming'])
    negative_texts = ' '.join(df[df['sentiment'] == 'negative']['stemming'])
    neutral_texts = ' '.join(df[df['sentiment'] == 'neutral']['stemming'])
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    
    def plot_wc(ax, text, title, cmap):
        if len(text.strip()) > 0:
            wc = WordCloud(width=600, height=400, background_color='white', colormap=cmap, max_words=100).generate(text)
            ax.imshow(wc, interpolation='bilinear')
            ax.set_title(title, fontsize=16)
            ax.axis('off')
        else:
            ax.text(0.5, 0.5, 'Tidak ada data', ha='center', va='center', fontsize=14)
            ax.axis('off')

    plot_wc(axes[0], positive_texts, 'WordCloud - Sentimen POSITIF', 'Greens')
    plot_wc(axes[1], negative_texts, 'WordCloud - Sentimen NEGATIF', 'Reds')
    plot_wc(axes[2], neutral_texts, 'WordCloud - Sentimen NETRAL', 'Greys')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUTS_DIR, 'wordclouds.png'))

def plot_top_words(df):
    positive_texts = ' '.join(df[df['sentiment'] == 'positive']['stemming'])
    negative_texts = ' '.join(df[df['sentiment'] == 'negative']['stemming'])
    neutral_texts = ' '.join(df[df['sentiment'] == 'neutral']['stemming'])

    def get_top_words(text, n=15):
        if not text or len(text.strip()) == 0: return []
        words = text.split()
        words = [w for w in words if len(w) > 3]
        return Counter(words).most_common(n)

    top_pos = get_top_words(positive_texts)
    top_neg = get_top_words(negative_texts)
    top_neu = get_top_words(neutral_texts)
    
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    
    def plot_bar(ax, data, title, color):
        if data:
            words, counts = zip(*data)
            ax.barh(words, counts, color=color, alpha=0.8)
            ax.set_title(title, fontsize=14)
            ax.set_xlabel('Frekuensi', fontsize=12)
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Tidak ada data', ha='center', va='center', fontsize=12)
            ax.set_title(title, fontsize=14)

    plot_bar(axes[0], top_pos, 'Top 15 Kata - Sentimen POSITIF', '#2ecc71')
    plot_bar(axes[1], top_neg, 'Top 15 Kata - Sentimen NEGATIF', '#e74c3c')
    plot_bar(axes[2], top_neu, 'Top 15 Kata - Sentimen NETRAL', '#95a5a6')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUTS_DIR, 'top_words.png'))

def analyze_tfidf(df):
    print("Performing TF-IDF Analysis...")
    df_clean = df.dropna(subset=['stemming'])
    df_list = df_clean['stemming'].fillna('').tolist()
    
    if not df_list:
        print("No data for TF-IDF.")
        return

    tfidf_vectorizer = TfidfVectorizer()
    X = tfidf_vectorizer.fit_transform(df_list)
    
    feature_names = tfidf_vectorizer.get_feature_names_out()
    tfidf_sums = X.sum(axis=0)
    
    tfidf_df = pd.DataFrame({'term': feature_names, 'tfidf_sum': tfidf_sums.tolist()[0]})
    tfidf_df = tfidf_df.sort_values(by='tfidf_sum', ascending=False)
    
    print("\nTop 10 Terms by Total TF-IDF Score")
    print(tfidf_df.head(10))
    
    # Plot Top 20
    top_20_terms = tfidf_df.head(20)
    plt.figure(figsize=(8, 6))
    sns.barplot(x='tfidf_sum', y='term', data=top_20_terms, palette='viridis')
    plt.title('Top 20 Terms by Total TF-IDF Score', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUTS_DIR, 'tfidf_ranking.png'))
