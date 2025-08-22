import tempfile
from pathlib import Path
import json
import os

import streamlit as st

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from extractor import NewspaperEducationExtractor
from config import CONFIDENCE_THRESHOLD, KEYWORD_MIN_MATCH, NUM_WORKERS, IS_SPACES, SEMANTIC_THRESHOLD

st.set_page_config(
    page_title="Enhanced Newspaper Education Extractor",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("📰 Enhanced Newspaper Education Extractor")
    st.caption("🧠 Advanced AI-powered extraction with semantic analysis and intelligent filtering")
    
    if IS_SPACES:
        st.info("🤗 Running on Hugging Face Spaces with enhanced semantic features!")
    
    # Enhanced usage instructions
    with st.expander("ℹ️ Enhanced Features & How to Use", expanded=False):
        st.markdown("""
        ### 🚀 **Enhanced AI Features:**
        - **🎯 Smart Detection**: YOLO v8 for precise article boundary detection
        - **🧠 Semantic Analysis**: Advanced contextual understanding using Sentence-BERT
        - **📝 Intelligent OCR**: Deskewing and noise reduction for better text extraction
        - **🤖 AI Summarization**: Facebook BART for high-quality summaries
        - **🔍 Context Filtering**: Excludes irrelevant content automatically
        
        ### 📖 **How to Use:**
        1. **Upload** a clear newspaper PDF (max 15MB)
        2. **Adjust** semantic threshold and other settings in sidebar
        3. **Click Extract** to analyze with advanced AI
        4. **Explore** detailed results with semantic scores
        5. **Download** comprehensive JSON with analysis details
        
        ### 🎓 **Perfect for:**
        - Education researchers and policy analysts
        - Journalists tracking education trends
        - Academic content analysis
        - Media monitoring for education topics
        """)

    # Initialize session state
    if "results" not in st.session_state:
        st.session_state.results = None
    if "processing_complete" not in st.session_state:
        st.session_state.processing_complete = False

    # Enhanced sidebar with semantic controls
    with st.sidebar:
        st.header("⚙️ Enhanced Settings")
        
        # YOLO settings
        st.subheader("🎯 Detection Settings")
        conf_threshold = st.slider(
            "YOLO Confidence", 0.3, 0.95, 
            value=float(CONFIDENCE_THRESHOLD), step=0.01,
            help="Higher values = more precise article detection"
        )
        
        # Semantic settings
        st.subheader("🧠 Semantic Analysis")
        semantic_threshold = st.slider(
            "Semantic Threshold", 0.2, 0.8, 
            value=float(SEMANTIC_THRESHOLD), step=0.05,
            help="Higher values = stricter semantic matching"
        )
        min_keywords = st.slider(
            "Min Keywords", 1, 5, 
            value=int(KEYWORD_MIN_MATCH), step=1,
            help="Minimum education keywords required"
        )
        
        # Processing options
        st.subheader("🔧 Processing Options")
        save_crops = st.checkbox("Save article crops", value=False)
        show_analysis = st.checkbox("Show semantic analysis details", value=True)
        
        st.markdown("---")
        st.markdown("### 🚀 **AI Stack:**")
        st.markdown("• **YOLO v8** - Object Detection")
        st.markdown("• **Sentence-BERT** - Semantic Analysis") 
        st.markdown("• **Tesseract OCR** - Text Extraction")
        st.markdown("• **Facebook BART** - Summarization")
        st.markdown("• **scikit-learn** - Similarity Scoring")

    # File uploader
    max_size_mb = 15 if IS_SPACES else 25
    uploaded_pdf = st.file_uploader(
        f"📄 Upload Newspaper PDF (max {max_size_mb}MB)", 
        type=["pdf"],
        help="Select a clear, high-resolution newspaper PDF for best results"
    )

    # Enhanced file validation
    if uploaded_pdf is not None:
        file_size_mb = uploaded_pdf.size / (1024 * 1024)
        
        if file_size_mb > max_size_mb:
            st.error(f"📄 File too large: {file_size_mb:.1f}MB")
            st.error(f"🚫 Maximum allowed: {max_size_mb}MB for optimal processing")
            st.info("💡 **Optimization tips:** Compress PDF, reduce resolution, or split into smaller files")
            return
        elif file_size_mb > 10:
            st.warning(f"⚠️ Large file ({file_size_mb:.1f}MB) - enhanced processing may take 3-5 minutes")
        else:
            st.success(f"✅ File ready for AI analysis: {file_size_mb:.1f}MB")

    # Enhanced extraction button
    extract_button = st.button(
        "🧠 Extract with Enhanced AI", 
        type="primary", 
        disabled=uploaded_pdf is None,
        help="Start advanced semantic analysis and extraction"
    )

    # Enhanced processing
    if extract_button and uploaded_pdf is not None:
        st.session_state.results = None
        st.session_state.processing_complete = False
        
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_pdf.read())
            tmp_path = tmp.name

        # Initialize enhanced extractor
        with st.spinner("🔧 Loading enhanced AI models..."):
            try:
                extractor = NewspaperEducationExtractor(
                    min_keyword_matches=min_keywords,
                    confidence_threshold=conf_threshold,
                    num_workers=NUM_WORKERS,
                    save_crops=save_crops,
                )
                st.success("✅ AI models loaded successfully!")
            except Exception as e:
                st.error(f"❌ Failed to load AI models: {str(e)}")
                return

        # Enhanced processing with detailed progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("📄 Converting PDF to high-quality images...")
            progress_bar.progress(15)
            
            status_text.text("🎯 Detecting article regions with YOLO AI...")
            progress_bar.progress(30)
            
            status_text.text("📝 Extracting text with enhanced OCR...")
            progress_bar.progress(50)
            
            status_text.text("🧠 Performing semantic analysis with Sentence-BERT...")
            progress_bar.progress(70)
            
            status_text.text("🤖 Generating AI summaries with BART...")
            progress_bar.progress(85)
            
            # Process with enhanced features
            results = extractor.process_newspaper(tmp_path)
            
            progress_bar.progress(100)
            status_text.text("✅ Enhanced AI processing complete!")
            
            # Store results
            st.session_state.results = results
            st.session_state.processing_complete = True
            
            # Cleanup
            try:
                os.unlink(tmp_path)
            except:
                pass
            
            progress_bar.empty()
            status_text.empty()
            
        except Exception as e:
            st.error(f"❌ Enhanced processing failed: {str(e)}")
            st.info("Try with a clearer PDF or adjust the semantic threshold")
            progress_bar.empty()
            status_text.empty()
            try:
                os.unlink(tmp_path)
            except:
                pass

    # Enhanced results display
    if st.session_state.results is not None and st.session_state.processing_complete:
        results = st.session_state.results
        stats = results.get("processing_stats", {})
        semantic_summary = results.get("semantic_analysis_summary", {})
        
        # Enhanced summary metrics
        st.subheader("📊 Enhanced Analysis Results")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("📄 Pages", stats.get("total_pages", 0))
        col2.metric("🔍 Detected", stats.get("total_articles_detected", 0))
        col3.metric("🎓 Education", stats.get("education_articles_found", 0))
        col4.metric("🧠 Semantic", "✅" if results.get("semantic_enabled", False) else "❌")
        col5.metric("📈 Avg Score", f"{semantic_summary.get('average_semantic_score', 0):.3f}")

        # Semantic analysis overview
        if show_analysis and semantic_summary:
            with st.expander("🧠 Detailed Semantic Analysis", expanded=True):
                col1, col2, col3 = st.columns(3)
                col1.metric("Articles Analyzed", semantic_summary.get('total_articles_analyzed', 0))
                col2.metric("Core Keyword Matches", semantic_summary.get('core_keyword_matches', 0))
                col3.metric("Semantic Filtering", "Enabled" if semantic_summary.get('semantic_filtering_enabled', False) else "Disabled")

        # Enhanced education articles display
        articles = results.get("education_articles", [])
        
        if articles:
            st.subheader(f"🎓 Education Articles Found ({len(articles)})")
            
            # Enhanced filtering
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            with filter_col1:
                all_keywords = sorted(set(kw for article in articles for kw in article.get('keywords_found', [])))
                keyword_filter = st.selectbox("🔍 Filter by Keyword", ["All"] + all_keywords, key="kw_filter")
            
            with filter_col2:
                min_confidence = st.slider("📊 Min Detection Confidence", 0.0, 1.0, 0.0, 0.05, key="conf_filter")
            
            with filter_col3:
                min_semantic = st.slider("🧠 Min Semantic Score", 0.0, 1.0, 0.0, 0.05, key="sem_filter")
            
            # Apply enhanced filters
            filtered_articles = articles
            if keyword_filter != "All":
                filtered_articles = [a for a in articles if keyword_filter in a.get('keywords_found', [])]
            if min_confidence > 0:
                filtered_articles = [a for a in articles if a.get('confidence', 0) >= min_confidence]
            if min_semantic > 0:
                filtered_articles = [a for a in articles if a.get('semantic_analysis', {}).get('semantic_score', 0) >= min_semantic]
            
            if not filtered_articles:
                st.info("🔍 No articles match your filter criteria. Try adjusting the filters.")
            
            # Enhanced article display
            for i, article in enumerate(filtered_articles, 1):
                confidence = article.get('confidence', 0)
                semantic_analysis = article.get('semantic_analysis', {})
                semantic_score = semantic_analysis.get('semantic_score', 0)
                
                # Dynamic confidence indicators
                conf_emoji = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🔴"
                sem_emoji = "🧠" if semantic_score > 0.5 else "🤔" if semantic_score > 0.3 else "🔍"
                
                with st.expander(f"{conf_emoji}{sem_emoji} Article {i} - Page {article['page']} (Det: {confidence:.2f}, Sem: {semantic_score:.3f})"):
                    # Enhanced metadata display
                    meta_col1, meta_col2, meta_col3 = st.columns(3)
                    meta_col1.write(f"**🏷️ Keywords:** {', '.join(article.get('keywords_found', [])[:5])}")
                    meta_col1.write(f"**📍 Location:** Page {article['page']}")
                    
                    meta_col2.write(f"**📝 Text Length:** {article.get('text_length', 0)} chars")
                    meta_col2.write(f"**🎯 Detection Conf:** {confidence:.3f}")
                    
                    meta_col3.write(f"**🧠 Semantic Score:** {semantic_score:.3f}")
                    meta_col3.write(f"**🔍 Core Keywords:** {'✅' if semantic_analysis.get('has_core_keyword', False) else '❌'}")
                    
                    # Detailed semantic analysis
                    if show_analysis and semantic_analysis:
                        with st.expander("🔬 Semantic Analysis Details"):
                            analysis_col1, analysis_col2 = st.columns(2)
                            analysis_col1.write(f"**Criteria Met:** {semantic_analysis.get('criteria_met', 0)}/4")
                            analysis_col1.write(f"**Context Score:** {semantic_analysis.get('context_score', 0):.3f}")
                            analysis_col2.write(f"**Keyword Count:** {semantic_analysis.get('keyword_count', 0)}")
                            analysis_col2.write(f"**Enhanced Processing:** {'✅' if article.get('enhanced_processing', False) else '❌'}")
                    
                    # Enhanced AI Summary
                    st.markdown("**🤖 AI-Generated Summary:**")
                    summary = article.get("summary", "No summary available")
                    if summary and len(summary) > 10:
                        st.write(summary)
                        
                        # Summary quality indicator
                        if len(summary) > 100:
                            st.caption("📊 High-quality summary generated")
                        elif len(summary) > 50:
                            st.caption("📊 Standard summary generated")
                        else:
                            st.caption("📊 Brief summary generated")
                    else:
                        st.info("🤖 No summary could be generated for this article")
                    
                    # Full text with better formatting
                    with st.expander("📄 View Full Extracted Text"):
                        full_text = article.get("full_text", "No text extracted")
                        if full_text and len(full_text) > 20:
                            st.text_area("", full_text, height=200, key=f"enhanced_text_{i}")
                            st.caption(f"📝 Text quality: {'High' if len(full_text) > 500 else 'Medium' if len(full_text) > 200 else 'Basic'}")
                        else:
                            st.info("📝 No readable text could be extracted from this article")
        else:
            st.info("🔍 No education articles found with current settings")
            st.info("**Try adjusting:**")
            st.info("• Lower the semantic threshold")
            st.info("• Reduce minimum keywords requirement") 
            st.info("• Use a PDF with clearer education content")

        # Enhanced download section
        st.subheader("💾 Download Enhanced Results")
        
        col1, col2 = st.columns(2)
        with col1:
            # Full results with semantic analysis
            json_data = json.dumps(results, indent=2, ensure_ascii=False).encode("utf-8")
            filename = f"enhanced_education_analysis_{uploaded_pdf.name}.json"
            
            st.download_button(
                "📥 Download Complete Analysis (JSON)",
                data=json_data,
                file_name=filename,
                mime="application/json",
                help="Includes all semantic analysis details and scores"
            )
        
        with col2:
            # Simplified results for basic use
            simplified_results = {
                "summary": stats,
                "articles": [
                    {
                        "page": art["page"],
                        "keywords": art.get("keywords_found", []),
                        "summary": art.get("summary", ""),
                        "confidence": art.get("confidence", 0),
                        "semantic_score": art.get("semantic_analysis", {}).get("semantic_score", 0)
                    }
                    for art in articles
                ]
            }
            
            simple_json = json.dumps(simplified_results, indent=2, ensure_ascii=False).encode("utf-8")
            simple_filename = f"simple_results_{uploaded_pdf.name}.json"
            
            st.download_button(
                "📋 Download Simple Results (JSON)",
                data=simple_json,
                file_name=simple_filename,
                mime="application/json",
                help="Simplified results without detailed analysis"
            )
        
        # Reset button
        if st.button("🔄 Analyze Another PDF"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]

    # Enhanced footer
    st.markdown("---")
    st.markdown("### 🚀 Enhanced with Advanced AI")
    st.markdown("**Semantic Analysis** • **Intelligent Filtering** • **Context Understanding** • **Quality Summarization**")
    st.markdown("Made with ❤️ using Streamlit • Powered by 🤗 Hugging Face • Enhanced AI Processing")

if __name__ == "__main__":
    main()
