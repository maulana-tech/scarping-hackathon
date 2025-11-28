import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from . import config

# Download NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def remove_punctuation(text):
    if not isinstance(text, str): return ""
    text = re.sub(r"[^\x00-\x7f]", r"", text)
    text = re.sub(r"(\\u[0-9A-Fa-f]+)", r"", text)
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    return text.translate(str.maketrans("","",string.punctuation))

def remove_number(text):
    if not isinstance(text, str): return ""
    text = re.sub(r"\b\d+\b", " ", text)
    return text.translate(str.maketrans("","",string.punctuation))

def remove_single_char(text):
    if not isinstance(text, str): return ""
    text = re.sub(r"\b[a-zA-Z]\b", "", text)
    return text.translate(str.maketrans("","",string.punctuation))

def remove_multiple_whitespace(text):
    if not isinstance(text, str): return ""
    text = re.sub('\s+',' ',text)
    return text.translate(str.maketrans("","",string.punctuation))

def case_folding(text):
    if isinstance(text, str):
        return text.lower()
    return text

def load_kamus_baku():
    try:
        kamus_data = pd.read_excel(config.KAMUS_BAKU_FILE)
        return dict(zip(kamus_data['tidak_baku'], kamus_data['kata_baku']))
    except FileNotFoundError:
        print(f"Warning: Kamus baku file not found at {config.KAMUS_BAKU_FILE}. Skipping normalization.")
        return {}

def replace_taboo_words(text, kamus_tidak_baku):
    if isinstance(text, str):
        words = text.split()
        replaced_words = []
        for word in words:
            if word in kamus_tidak_baku:
                baku_word = kamus_tidak_baku[word]
                if isinstance(baku_word, str) and all(char.isalpha() or char.isspace() for char in baku_word):
                    replaced_words.append(baku_word)
                else:
                    replaced_words.append(word) # Fallback if baku_word is invalid
            else:
                replaced_words.append(word)
        return ' '.join(replaced_words)
    return ' '

def tokenize(text):
    if isinstance(text, str):
        return text.split()
    return []

def get_stopwords():
    stop_words = stopwords.words('indonesian')
    new_stopwords = ["masingmasing", "benarbenar","dgn","gak","ga","ya","nya","yg",
                     "tolong","gabisa","mohon","aja","ngga","banget","udah","nggak",
                     "gimana","ini","gk","sih","dan","saya","karena","di","ke","untuk"]
    special_words = ["terima","kasih","terimakasih","apk"]
    stop_words.extend(new_stopwords)
    stop_words.extend(special_words)
    return set(stop_words)

def remove_stopwords_func(tokens, stop_words):
    return [word for word in tokens if word not in stop_words]

def get_stemmer():
    factory = StemmerFactory()
    return factory.create_stemmer()

def preprocess_dataframe(df):
    print("Starting preprocessing...")
    
    # 1. Cleaning
    print("Cleaning text...")
    df["cleaning"] = df['content'].apply(remove_punctuation)
    df["cleaning"] = df["cleaning"].apply(remove_number)
    df["cleaning"] = df["cleaning"].apply(remove_single_char)
    df["cleaning"] = df["cleaning"].apply(remove_multiple_whitespace)
    
    # 2. Case Folding
    print("Case folding...")
    df['casefolding'] = df['cleaning'].apply(case_folding)
    
    # 3. Normalization
    print("Normalizing words...")
    kamus_tidak_baku = load_kamus_baku()
    df['hasil normalisasi'] = df['casefolding'].apply(lambda x: replace_taboo_words(x, kamus_tidak_baku))
    
    # 4. Tokenizing
    print("Tokenizing...")
    df['tokenize'] = df['hasil normalisasi'].apply(tokenize)
    
    # 5. Stopword Removal
    print("Removing stopwords...")
    stop_words = get_stopwords()
    df['stopword removal'] = df['tokenize'].apply(lambda x: remove_stopwords_func(x, stop_words))
    
    # 6. Stemming
    print("Stemming (this may take a while)...")
    stemmer = get_stemmer()
    
    # Optimization: Stem unique terms first
    all_tokens = [term for sublist in df['stopword removal'] for term in sublist]
    unique_terms = list(set(all_tokens))
    term_dict = {term: stemmer.stem(term) for term in unique_terms}
    
    def apply_stemming(tokens):
        return [term_dict[t] for t in tokens if t in term_dict]

    df['stemming'] = df['stopword removal'].apply(lambda x: ' '.join(apply_stemming(x)))
    
    print("Preprocessing complete.")
    return df
