from transformers import pipeline
import pandas as pd

def load_model():
    print("Loading RoBERTa model...")
    return pipeline(
        "text-classification",
        model="w11wo/indonesian-roberta-base-sentiment-classifier"
    )

def predict_sentiment(df):
    classifier = load_model()
    
    print("Predicting sentiment...")
    texts = [t for t in df["hasil normalisasi"].tolist() if isinstance(t, str) and t.strip() != ""]
    
    # Batch processing could be added here for efficiency, but keeping it simple as per original
    results = classifier(texts)
    
    # Map results back to DataFrame
    # Initialize columns
    df["sentiment"] = None
    df["score"] = None
    
    # Create a mapping dictionary for faster lookup if texts are unique, 
    # but since there might be duplicates, we iterate carefully.
    # The original code logic:
    # df.loc[df["hasil normalisasi"].isin(texts), "sentiment"] = [r["label"] for r in results]
    # This logic in original code has a potential bug if multiple rows have same text but different context (unlikely here)
    # or if the order isn't preserved perfectly. 
    # However, we will stick to a safer iteration method.
    
    results_iter = iter(results)
    
    sentiments = []
    scores = []
    
    for text in df["hasil normalisasi"]:
        if isinstance(text, str) and text.strip() != "":
            res = next(results_iter)
            sentiments.append(res["label"])
            scores.append(res["score"])
        else:
            sentiments.append(None)
            scores.append(None)
            
    df["sentiment"] = sentiments
    df["score"] = scores
    
    return df
