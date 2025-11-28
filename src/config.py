import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')

# Input Files (Mapped to available files in data/)
FILE_PLAYSTORE = os.path.join(DATA_DIR, 'Data-Scrape-PlayStore.csv') 
FILE_YOUTUBE = os.path.join(DATA_DIR, 'Data-Scrape-YouTube.csv')
FILE_TWITTER_TIKTOK = os.path.join(DATA_DIR, 'Data-Combined-Twitter-Tiktok.csv')

# Auxiliary Files
KAMUS_BAKU_FILE = os.path.join(DATA_DIR, 'kamuskatabaku.xlsx')

# Output Files
OUTPUT_PREPROCESSED_CSV = os.path.join(PROCESSED_DATA_DIR, 'CoreTax Preprocessing Results.csv')
OUTPUT_BERTOPIC_CSV = os.path.join(PROCESSED_DATA_DIR, 'BERTopic-CoreTax-data.csv')
OUTPUT_BERTOPIC_MODEL = os.path.join(MODELS_DIR, 'bertopic_coretax_model')
