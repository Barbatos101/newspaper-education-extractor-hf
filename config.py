import os
from pathlib import Path
import requests
import logging

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Detect if running on Hugging Face Spaces
IS_SPACES = os.getenv("SPACE_ID") is not None

# Project paths
PROJECT_ROOT = Path(__file__).parent
MODEL_PATH = PROJECT_ROOT / "models" / "bestmodel.pt"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Create directories
for d in [OUTPUT_DIR / "images", OUTPUT_DIR / "crops", OUTPUT_DIR / "results"]:
    d.mkdir(parents=True, exist_ok=True)

def ensure_model_downloaded():
    """Download model from GitHub Release if not present"""
    if not MODEL_PATH.exists():
        print("Downloading YOLO model from GitHub Release...")
        url = "https://github.com/Barbatos101/newspaper-education-extractor-hf/releases/download/v1.0/bestmodel.pt"
        
        try:
            MODEL_PATH.parent.mkdir(exist_ok=True)
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            with open(MODEL_PATH, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\rDownload progress: {percent:.1f}%", end='')
            print(f"\nModel downloaded to {MODEL_PATH}")
        except Exception as e:
            print(f"Error downloading model: {e}")
            raise

# Download model on import
ensure_model_downloaded()

# Optimized settings for Hugging Face Spaces
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.78'))
KEYWORD_MIN_MATCH = int(os.getenv('KEYWORD_MIN_MATCH', '2'))

# Enhanced education keywords for semantic filtering
EDUCATION_KEYWORDS = [
    'school', 'schools', 'education', 'educational',
    'student', 'students', 'teacher', 'teachers',
    'university', 'college', 'academic', 'classroom',
    'curriculum', 'exam', 'exams', 'graduation',
    'scholarship', 'principal', 'kindergarten', 'elementary',
    'secondary', 'admission', 'enrollment', 'faculty', 'campus',
    'homework', 'textbook', 'library', 'semester', 'grade',
    'syllabus', 'tuition', 'institute', 'staff', 'board',
    'lesson', 'lessons', 'instructor', 'pupil', 'pupils', 
    'academy', 'preschool', 'high school', 'primary'
]

# OCR settings
OCR_LANG = "eng"
OCR_PSM_PRIMARY = 6
OCR_PSM_FALLBACK = 4

# Threading - optimized for Spaces
NUM_WORKERS = 1

# Enhanced LLM settings with semantic analysis
SUMMARIZATION_MODEL = "facebook/bart-large-cnn"
MAX_SUMMARY_LENGTH = 120
MAX_INPUT_CHARS_FOR_SUMMARY = 1200

# Performance optimization for Spaces
REDUCED_DPI = 150 if IS_SPACES else 200
SEMANTIC_THRESHOLD = 0.35
SEMANTIC_MODEL = "all-MiniLM-L6-v2"

# Semantic context patterns
EDUCATION_CONTEXTS = [
    "school education", "student performance", "teacher training",
    "educational system", "learning outcomes", "curriculum development",
    "academic achievement", "classroom instruction", "school administration",
    "educational policy", "student assessment", "language policy"
]

# Context exclusions for better filtering
CONTEXT_EXCLUSIONS = [
    'weather', 'temperature', 'heat', 'celsius', 'fahrenheit',
    'clinical study', 'medical study', 'market research', 'data analysis',
    'laboratory experiment', 'scientific research', 'thermal', 'climate'
]
