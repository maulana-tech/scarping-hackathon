import pandas as pd
import os
from . import config
from . import data_loader
from . import preprocessing
from . import sentiment_analysis
from . import visualization
from . import topic_modeling

def main():
    print("=== CoreTax Sentiment Analysis Pipeline ===")
    
    # 1. Load Data
    df = data_loader.load_and_merge_data()
    
    # 2. Preprocessing
    df = preprocessing.preprocess_dataframe(df)
    
    # 3. Sentiment Analysis
    df = sentiment_analysis.predict_sentiment(df)
    
    # Save Preprocessed Results
    print(f"Saving preprocessed data to {config.OUTPUT_PREPROCESSED_CSV}")
    df.to_csv(config.OUTPUT_PREPROCESSED_CSV, index=False)
    
    # 4. Visualization
    print("Generating visualizations...")
    visualization.plot_sentiment_distribution(df)
    visualization.plot_sentiment_by_source(df)
    visualization.generate_wordclouds(df)
    visualization.plot_top_words(df)
    visualization.analyze_tfidf(df)
    
    # 5. Topic Modeling (BERTopic)
    topic_modeling.run_topic_modeling(df)
    
    print("=== Pipeline Completed Successfully ===")

if __name__ == "__main__":
    main()
