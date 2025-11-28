from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import pandas as pd
import os
from . import config

def run_topic_modeling(df):
    print("Running BERTopic modeling...")
    
    neg_data = df[df['sentiment'] == 'negative']
    texts = neg_data['hasil normalisasi'].fillna('').tolist()
    
    if not texts:
        print("No negative sentiment data for topic modeling.")
        return

    embedding_model = SentenceTransformer("distiluse-base-multilingual-cased-v2")
    
    topic_model = BERTopic(
        language="indonesian",
        embedding_model=embedding_model,
        n_gram_range=(1, 2),
        min_topic_size=20
    )
    
    topics, probs = topic_model.fit_transform(texts)
    print(topic_model.get_topic_info())
    
    # Save model
    topic_model.save(config.OUTPUT_BERTOPIC_MODEL)
    
    # Create results DataFrame
    df_results = pd.DataFrame({
        "text": texts,
        'sentiment': neg_data['sentiment'].values,
        "topic": topics,
        "probability": probs,
        "topic_name": [topic_model.get_topic(t) for t in topics],
        "topic_words": [topic_model.get_topic(t)[0] if topic_model.get_topic(t) else "" for t in topics],
        "source": neg_data['source'].values
    })
    
    # Save results
    df_results.to_csv(config.OUTPUT_BERTOPIC_CSV, index=False)
    print(f"BERTopic results saved to {config.OUTPUT_BERTOPIC_CSV}")
    
    # Visualizations (Saved as HTML)
    topic_model.visualize_topics().write_html(os.path.join(config.OUTPUTS_DIR, "bertopic_topics.html"))
    topic_model.visualize_barchart().write_html(os.path.join(config.OUTPUTS_DIR, "bertopic_barchart.html"))
    topic_model.visualize_hierarchy().write_html(os.path.join(config.OUTPUTS_DIR, "bertopic_hierarchy.html"))
