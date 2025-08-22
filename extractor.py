import os
import cv2
import fitz
import numpy as np
import pytesseract
import json
import re
import logging
from pathlib import Path
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import threading

from config import *

# Set environment variables for optimization
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def crop_and_deskew(img):
    """Advanced image preprocessing with deskewing"""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
        
        # Detect skew angle
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        angles = []
        if lines is not None:
            for rho, theta in lines[:10]:
                angle = np.degrees(theta) - 90
                if abs(angle) < 45:
                    angles.append(angle)
        
        if angles:
            skew_angle = np.median(angles)
            if abs(skew_angle) > 0.5:
                (h, w) = gray.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
                deskewed = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                return deskewed
        
        return gray
    except Exception:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()

def clean_ocr_text(text):
    """Advanced OCR text cleaning"""
    if not text:
        return ""
    
    try:
        # Remove line artifacts and broken characters
        text = re.sub(r'[|\\/=↔_•¤©®™]+', '', text)
        text = re.sub(r'[^\w\s.,!?:;\'\"()\-\n]', ' ', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s{2,}', ' ', text)
        text = re.sub(r'-\s*\n\s*', '', text)
        text = re.sub(r'-\s+', ' ', text)
        
        # Clean up lines
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if len(line) > 10 and re.search(r'[a-zA-Z]', line):
                alpha_ratio = len(re.findall(r'[a-zA-Z]', line)) / len(line) if line else 0
                if alpha_ratio > 0.3:
                    cleaned_lines.append(line)
        
        text = ' '.join(cleaned_lines)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    except Exception:
        return str(text).strip() if text else ""

class SemanticEducationFilter:
    """Enhanced semantic education filtering using sentence transformers"""
    
    def __init__(self, model_name: str = None):
        self.logger = logging.getLogger(__name__)
        model_name = model_name or SEMANTIC_MODEL
        
        try:
            self.sentence_model = SentenceTransformer(model_name)
            self.semantic_available = True
            self.logger.info(f"Loaded semantic model: {model_name}")
        except Exception as e:
            self.logger.error(f"Semantic model loading failed: {e}")
            self.sentence_model = None
            self.semantic_available = False

        self.education_keywords = EDUCATION_KEYWORDS
        self.context_exclusions = CONTEXT_EXCLUSIONS
        self.core_keywords = [
            'school', 'college', 'university', 'teacher', 'student',
            'classroom', 'exam', 'curriculum', 'faculty', 'principal',
            'admission', 'syllabus', 'education', 'educational'
        ]
        
        if self.semantic_available:
            try:
                self.keyword_embeddings = self.sentence_model.encode(self.education_keywords)
                self.education_contexts = EDUCATION_CONTEXTS
                self.context_embeddings = self.sentence_model.encode(self.education_contexts)
                self.logger.info("Created education keyword and context embeddings")
            except Exception as e:
                self.logger.error(f"Failed to create embeddings: {e}")
                self.semantic_available = False

    def is_education_article(self, text: str, min_keywords: int = 2, semantic_threshold: float = None) -> Tuple[bool, List[str], Dict]:
        """Enhanced semantic analysis with detailed scoring"""
        
        if not text or len(text.strip()) < 30:
            return False, [], {}
        
        semantic_threshold = semantic_threshold or SEMANTIC_THRESHOLD
        text_lower = text.lower()
        
        # Check for exclusion contexts first
        for exclusion in self.context_exclusions:
            if exclusion in text_lower:
                return False, [], {"exclusion_found": exclusion}
        
        # Traditional keyword matching
        found_keywords = []
        for keyword in self.education_keywords:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text_lower):
                found_keywords.append(keyword)
        
        # Require at least one core education keyword
        has_core_keyword = any(core in found_keywords for core in self.core_keywords)
        
        # Semantic similarity analysis
        semantic_score = 0.0
        context_score = 0.0
        analysis_details = {
            "keyword_count": len(found_keywords),
            "has_core_keyword": has_core_keyword,
            "semantic_available": self.semantic_available
        }
        
        if self.semantic_available and self.sentence_model:
            try:
                text_embedding = self.sentence_model.encode([text])
                keyword_similarities = cosine_similarity(text_embedding, self.keyword_embeddings)
                semantic_score = np.max(keyword_similarities[0])
                
                context_similarities = cosine_similarity(text_embedding, self.context_embeddings)
                context_score = np.max(context_similarities)
                
                analysis_details.update({
                    "semantic_score": float(semantic_score),
                    "context_score": float(context_score)
                })
            except Exception as e:
                self.logger.warning(f"Semantic analysis failed: {e}")
        
        # Enhanced decision making
        criteria_met = 0
        
        # Keyword-based criteria
        if len(found_keywords) >= min_keywords and has_core_keyword:
            criteria_met += 2
        elif len(found_keywords) >= min_keywords:
            criteria_met += 1
        
        # Semantic criteria
        if semantic_score >= semantic_threshold:
            criteria_met += 1
        
        if context_score >= semantic_threshold:
            criteria_met += 1
        
        # Final decision
        if self.semantic_available:
            is_education = criteria_met >= 2
        else:
            is_education = len(found_keywords) >= min_keywords and has_core_keyword
        
        analysis_details["criteria_met"] = criteria_met
        analysis_details["is_education"] = is_education
        
        return is_education, found_keywords, analysis_details

class NewspaperEducationExtractor:
    def __init__(
        self,
        min_keyword_matches: Optional[int] = None,
        confidence_threshold: Optional[float] = None,
        summarization_model: Optional[str] = None,
        num_workers: Optional[int] = None,
        save_crops: bool = False,
    ):
        """Initialize the enhanced extractor with advanced semantic filtering"""
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Configure OpenCV
        cv2.setNumThreads(0)
        try:
            cv2.ocl.setUseOpenCL(False)
        except:
            pass

        # Runtime settings
        self.keyword_min_match = min_keyword_matches or KEYWORD_MIN_MATCH
        self.confidence_threshold = confidence_threshold or CONFIDENCE_THRESHOLD
        self.num_workers = num_workers or NUM_WORKERS
        self.save_crops = save_crops
        
        # Threading locks
        self._ocr_lock = threading.Lock()
        self._summ_lock = threading.Lock()

        # Load YOLO model
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        
        self.yolo_model = YOLO(str(MODEL_PATH))
        self.education_filter = SemanticEducationFilter()
        
        self.logger.info("Loaded YOLO model and enhanced semantic filter")
        
        # Initialize summarization
        self.logger.info("Loading summarization model...")
        try:
            model_name = summarization_model or SUMMARIZATION_MODEL
            self.summarizer = pipeline(
                "summarization",
                model=model_name,
                tokenizer=model_name,
                device=0 if torch.cuda.is_available() else -1
            )
            self.logger.info(f"Summarizer loaded: {model_name}")
        except Exception as e:
            self.logger.warning(f"Summarization model failed to load: {e}")
            self.summarizer = None

        self.logger.info("Enhanced extractor initialized successfully")

    def process_newspaper(self, pdf_path: str) -> Dict:
        """Complete processing pipeline with enhanced semantic analysis"""
        self.logger.info(f"Processing newspaper with enhanced semantic features: {pdf_path}")
        
        # Convert PDF to images
        image_paths = self.pdf_to_images(pdf_path)
        
        # Initialize results
        education_articles = []
        all_analysis_details = []
        stats = {
            'total_pages': len(image_paths),
            'total_articles_detected': 0,
            'education_articles_found': 0,
            'semantic_enabled': self.education_filter.semantic_available
        }
        
        # Process each page
        for page_num, image_path in enumerate(image_paths, 1):
            self.logger.info(f"Processing page {page_num}/{len(image_paths)}")
            
            # Detect articles
            articles = self.detect_articles(image_path, page_num)
            stats['total_articles_detected'] += len(articles)
            
            # Process articles with threading
            if articles:
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    future_to_idx = {
                        executor.submit(self._process_single_article, article, page_num): idx
                        for idx, article in enumerate(articles)
                    }
                    
                    for future in as_completed(future_to_idx):
                        try:
                            result = future.result(timeout=180)
                            if result:
                                education_articles.append(result)
                                stats['education_articles_found'] += 1
                                
                                # Log detailed analysis
                                analysis = result.get('semantic_analysis', {})
                                all_analysis_details.append(analysis)
                                
                                self.logger.info(
                                    f"Found education article: Page {result['page']}, "
                                    f"Article {result['article_id']}, "
                                    f"Semantic Score: {analysis.get('semantic_score', 0):.3f}, "
                                    f"Keywords: {result['keywords_found'][:3]}"
                                )
                        except Exception as e:
                            self.logger.warning(f"Article processing error: {e}")
        
        # Compile final results with enhanced analytics
        results = {
            'pdf_path': pdf_path,
            'processing_stats': stats,
            'education_articles': education_articles,
            'semantic_analysis_summary': {
                'total_articles_analyzed': len(all_analysis_details),
                'semantic_filtering_enabled': self.education_filter.semantic_available,
                'average_semantic_score': np.mean([a.get('semantic_score', 0) for a in all_analysis_details]) if all_analysis_details else 0,
                'core_keyword_matches': sum(1 for a in all_analysis_details if a.get('has_core_keyword', False))
            },
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'semantic_enabled': self.education_filter.semantic_available,
            'summarization_model': SUMMARIZATION_MODEL,
            'enhanced_features': True
        }
        
        try:
            pdf_name = Path(pdf_path).stem
            results_file = OUTPUT_DIR / "results" / f"{pdf_name}_education_articles_enhanced.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Enhanced processing complete! Found {len(education_articles)} education articles")
            self.logger.info(f"Results saved to: {results_file}")
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
        
        return results

    def pdf_to_images(self, pdf_path: str, dpi: int = None) -> List[str]:
        """Convert PDF pages to images with optimized DPI"""
        if dpi is None:
            dpi = REDUCED_DPI
            
        self.logger.info(f"Converting PDF at {dpi} DPI")
        
        try:
            pdf_document = fitz.open(pdf_path)
            image_paths = []
            pdf_name = Path(pdf_path).stem
            
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                mat = fitz.Matrix(dpi/72, dpi/72)
                pix = page.get_pixmap(matrix=mat)
                
                image_filename = f"{pdf_name}_page_{page_num + 1}.png"
                image_path = OUTPUT_DIR / "images" / image_filename
                pix.save(str(image_path))
                image_paths.append(str(image_path))
            
            pdf_document.close()
            return image_paths
        except Exception as e:
            self.logger.error(f"PDF conversion error: {e}")
            return []

    def detect_articles(self, image_path: str, page_num: int) -> List[Dict]:
        """Detect articles using YOLO with enhanced filtering"""
        try:
            results = self.yolo_model.predict(
                source=image_path,
                conf=self.confidence_threshold,
                imgsz=640,
                verbose=False,
                save=False
            )
            
            articles = []
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    
                    for i, (box, conf) in enumerate(zip(boxes, confidences)):
                        x1, y1, x2, y2 = map(int, box)
                        area = (x2 - x1) * (y2 - y1)
                        
                        # Enhanced area filtering
                        if area > 8000:  # Slightly higher threshold for better quality
                            articles.append({
                                'article_id': i + 1,
                                'bbox': [x1, y1, x2, y2],
                                'confidence': float(conf),
                                'area': area,
                                'image_path': image_path,
                                'page': page_num
                            })
            
            return articles
        except Exception as e:
            self.logger.error(f"Article detection error: {e}")
            return []

    def _process_single_article(self, article: Dict, page_num: int) -> Optional[Dict]:
        """Process one article with enhanced semantic analysis"""
        try:
            # Extract text using enhanced OCR
            crop_path, text = self.extract_article_crop_and_text(article)
            
            if len(text.strip()) < 50:  # Slightly higher threshold
                return None
            
            # Enhanced education filtering with detailed analysis
            is_education, keywords, analysis_details = self.education_filter.is_education_article(
                text, 
                self.keyword_min_match,
                SEMANTIC_THRESHOLD
            )
            
            if not is_education:
                return None
            
            # Generate enhanced summary
            summary = self.summarize_text(text)
            
            return {
                'page': page_num,
                'article_id': article['article_id'],
                'confidence': article['confidence'],
                'bbox': article['bbox'],
                'keywords_found': keywords,
                'full_text': text,
                'summary': summary,
                'crop_path': crop_path,
                'text_length': len(text),
                'semantic_analysis': analysis_details,
                'enhanced_processing': True
            }
        except Exception as e:
            self.logger.error(f"Article processing error: {e}")
            return None

    def extract_article_crop_and_text(self, article_data: Dict) -> Tuple[str, str]:
        """Extract article crop and perform enhanced OCR"""
        try:
            img = cv2.imread(article_data['image_path'])
            if img is None:
                return "", ""
            
            x1, y1, x2, y2 = article_data['bbox']
            crop = img[y1:y2, x1:x2]
            
            if crop.size == 0:
                return "", ""

            crop_path_str = ""
            if self.save_crops:
                try:
                    image_name = Path(article_data['image_path']).stem
                    crop_filename = f"{image_name}_article_{article_data['article_id']}.jpg"
                    crop_path = OUTPUT_DIR / "crops" / crop_filename
                    cv2.imwrite(str(crop_path), crop)
                    crop_path_str = str(crop_path)
                except Exception:
                    pass

            # Enhanced OCR with preprocessing
            with self._ocr_lock:
                gray = crop_and_deskew(crop)
                
                # Try primary PSM first
                text = pytesseract.image_to_string(
                    gray, 
                    config=f'--oem 3 --psm {OCR_PSM_PRIMARY} -l {OCR_LANG}'
                )
                
                # Fallback to alternative PSM if poor results
                if len(clean_ocr_text(text)) < 30:
                    text = pytesseract.image_to_string(
                        gray, 
                        config=f'--oem 3 --psm {OCR_PSM_FALLBACK} -l {OCR_LANG}'
                    )
                
                text = clean_ocr_text(text)

            return crop_path_str, text if text else ""

        except Exception as e:
            self.logger.error(f"Enhanced OCR error: {e}")
            return "", ""

    def summarize_text(self, text: str) -> str:
        """Enhanced text summarization with better preprocessing"""
        try:
            if not text or len(text.strip()) < 60:
                return text if text else ""
            
            cleaned_text = clean_ocr_text(text)
            
            if not cleaned_text or len(cleaned_text.strip()) < 60:
                return cleaned_text if cleaned_text else ""
            
            if self.summarizer:
                try:
                    with self._summ_lock:
                        summary = self.summarizer(
                            cleaned_text[:MAX_INPUT_CHARS_FOR_SUMMARY],
                            max_length=MAX_SUMMARY_LENGTH,
                            min_length=50,
                            do_sample=False
                        )
                    return summary[0]['summary_text']
                except Exception as e:
                    self.logger.warning(f"Summarization error: {e}")
            
            # Enhanced fallback summary
            sentences = re.split(r'(?<=[.!?])\s+', cleaned_text)
            if len(sentences) >= 3:
                return '. '.join(sentences[:3]) + '.'
            elif len(sentences) >= 2:
                return '. '.join(sentences[:2]) + '.'
            else:
                return cleaned_text[:250] + "..." if len(cleaned_text) > 250 else cleaned_text
                
        except Exception as e:
            self.logger.error(f"Enhanced summarization error: {e}")
            return text[:250] + "..." if text and len(text) > 250 else (text if text else "")
