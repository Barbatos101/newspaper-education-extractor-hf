import tempfile
from pathlib import Path
import json
import os

import streamlit as st

# Set environment variables before importing other modules
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from extractor import NewspaperEducationExtractor
from config import CONFIDENCE_THRESHOLD, KEYWORD_MIN_MATCH, NUM_WORKERS

st.set_page_config(page_title="Newspaper Education Extractor", layout="wide")
st.title("Newspaper Education Extractor with Semantic Features")
st.caption("Upload a newspaper PDF to detect, OCR, and summarize education-related articles.")

# Health check for Cloud Run
if st.query_params.get("health") == "check":
    st.write("OK")
    st.stop()

def main():
    # Health check endpoint for Cloud Run
    if st.query_params.get("healthz") is not None:
        st.write("OK")
        st.stop()
    
    # Initialize session state ONCE
    if "results" not in st.session_state:
        st.session_state.results = None
    if "uploaded_file_name" not in st.session_state:
        st.session_state.uploaded_file_name = None
    if "processing_complete" not in st.session_state:
        st.session_state.processing_complete = False

    with st.sidebar:
        st.header("Settings")
        conf_threshold = st.slider("YOLO confidence threshold", 0.3, 0.95, value=0.78, step=0.01)  # Fixed default to 0.78
        min_keywords = st.slider("Min education keywords", 1, 5, value=int(KEYWORD_MIN_MATCH), step=1)
        workers = st.slider("Workers", 1, 8, value=int(NUM_WORKERS), step=1)
        save_crops = st.checkbox("Save cropped images", value=False)
        
        st.info("🧠 Semantic filtering enabled")
        st.info("📝 Using sshleifer/distilbart-cnn-12-6")

    # File uploader
    uploaded_pdf = st.file_uploader("Upload newspaper PDF", type=["pdf"], key="pdf_uploader")

    # Enhanced file size validation for Cloud Run
    if uploaded_pdf is not None:
        file_size_mb = uploaded_pdf.size / (1024 * 1024)
        file_size_bytes = uploaded_pdf.size
        
        # Cloud Run HTTP/1 limit is 32MB, use 25MB as safe limit
        max_size_mb = 25
        max_size_bytes = max_size_mb * 1024 * 1024
        
        if file_size_bytes > max_size_bytes:
            st.error(f"📄 File too large: {file_size_mb:.1f}MB")
            st.error(f"🚫 Maximum allowed: {max_size_mb}MB (Cloud Run limit)")
            st.info("💡 **Solutions:**")
            st.info("• Compress your PDF using online tools")
            st.info("• Split large PDFs into smaller sections")
            st.info("• Try PDFs with fewer pages or lower resolution")
            return
        elif file_size_bytes > 15 * 1024 * 1024:  # 15MB warning
            st.warning(f"⚠️ Large file ({file_size_mb:.1f}MB) - processing may take longer")
        else:
            st.success(f"✅ File uploaded successfully ({file_size_mb:.1f}MB)")
        
        st.session_state.uploaded_file_name = uploaded_pdf.name

    # Run extraction button - SIMPLIFIED LOGIC
    run_extraction = st.button("🚀 Run Extraction", type="primary", key="extract_btn")

    # Process when button is clicked - NO ST.RERUN CALLS
    if run_extraction and uploaded_pdf is not None:
        # Clear previous results
        st.session_state.results = None
        st.session_state.processing_complete = False
        
        # Write to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_pdf.read())
            tmp_path = tmp.name

        # Create extractor
        with st.spinner("🔧 Initializing AI models..."):
            try:
                extractor = NewspaperEducationExtractor(
                    min_keyword_matches=min_keywords,
                    confidence_threshold=conf_threshold,
                    num_workers=workers,
                    save_crops=save_crops,
                )
            except Exception as e:
                st.error(f"Failed to initialize extractor: {str(e)}")
                return

        # Processing with progress
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("📄 Converting PDF to images...")
                progress_bar.progress(20)
                
                status_text.text("🤖 Running YOLO detection...")
                progress_bar.progress(40)
                
                status_text.text("👁️ Performing OCR on detected articles...")
                progress_bar.progress(60)
                
                status_text.text("🧠 Applying semantic filtering...")
                progress_bar.progress(80)
                
                # Process the PDF
                results = extractor.process_newspaper(tmp_path)
                
                progress_bar.progress(100)
                status_text.text("✅ Processing complete!")
                
                # Store results in session state
                st.session_state.results = results
                st.session_state.processing_complete = True
                
                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
            except Exception as e:
                st.error(f"🚫 Processing failed: {str(e)}")
                st.info("This might be due to:")
                st.info("• File complexity or corruption")
                st.info("• Memory limitations")
                st.info("• Model loading issues")
                st.info("Try with a simpler or smaller PDF.")
                
                progress_bar.empty()
                status_text.empty()
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                return

    # Display results if available - OUTSIDE THE BUTTON LOGIC
    if st.session_state.results is not None and st.session_state.processing_complete:
        results = st.session_state.results
        
        # Display summary - FIXED
        stats = results.get("processing_stats", {})
        st.subheader("📊 Processing Summary")
        col1, col2, col3, col4 = st.columns(4)  # Unpack columns properly
        col1.metric("📄 Pages", stats.get("total_pages", 0))
        col2.metric("🔍 Detected", stats.get("total_articles_detected", 0))
        col3.metric("🎓 Education", stats.get("education_articles_found", 0))
        col4.metric("🧠 Semantic", "✅" if results.get("semantic_enabled", False) else "❌")

        # Show education articles
        articles = results.get("education_articles", [])
        if articles:
            st.subheader(f"🎓 Education Articles ({len(articles)} found)")
            
            # Filtering options
            filter_col1, filter_col2 = st.columns(2)
            with filter_col1:
                all_keywords = sorted(set(kw for article in articles for kw in article.get('keywords_found', [])))
                keyword_filter = st.selectbox(
                    "🔍 Filter by keyword:",
                    ["All"] + all_keywords,
                    index=0,
                    key="keyword_filter"
                )
            with filter_col2:
                min_confidence = st.slider("📊 Minimum confidence", 0.0, 1.0, 0.0, 0.05, key="conf_filter")
            
            # Apply filters
            filtered_articles = articles
            if keyword_filter != "All":
                filtered_articles = [a for a in articles if keyword_filter in a.get('keywords_found', [])]
            if min_confidence > 0:
                filtered_articles = [a for a in articles if a.get('confidence', 0) >= min_confidence]
            
            if not filtered_articles:
                st.info("No articles match your filter criteria. Try adjusting the filters.")
            
            # FIXED ARTICLE DISPLAY WITH PROPER COLUMN INDEXING AND ACCESSIBILITY
            for i, article in enumerate(filtered_articles, 1):
                confidence = article.get('confidence', 0)
                conf_color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🔴"
                
                with st.expander(f"{conf_color} {i}. Page {article['page']} • Article {article['article_id']} • conf={confidence:.2f}"):
                    # Metadata with FIXED column indexing
                    meta_cols = st.columns(3)
                    keywords = article.get('keywords_found', [])[:6]
                    meta_cols[0].write(f"**🏷️ Keywords:** {', '.join(keywords)}")
                    meta_cols[1].write(f"**📝 Text length:** {article.get('text_length', 0)} chars")
                    meta_cols[2].write(f"**📐 BBox:** {article.get('bbox', [])}")
                    
                    # DEBUG: Show crop path information
                    crop_path = article.get("crop_path")
                    if save_crops:
                        st.write(f"**Debug - Crop path:** {crop_path}")
                        if crop_path:
                            st.write(f"**Debug - File exists:** {Path(crop_path).exists()}")
                    
                    # Show crop if available
                    if article.get("crop_path") and Path(article["crop_path"]).exists():
                        st.image(str(article["crop_path"]), caption="🖼️ Article Crop", use_container_width=True)
                    elif save_crops:
                        st.info("🖼️ Crop was saved but image file not found")
                    
                    # Summary
                    st.markdown("**🤖 AI Summary**")
                    summary_text = article.get("summary", "No summary available")
                    if summary_text:
                        st.write(summary_text)
                    else:
                        st.info("No summary could be generated for this article.")
                    
                    # FIXED: Full text with proper label and accessibility
                    with st.expander("📄 View full OCR text"):
                        full_text = article.get("full_text", "No text extracted")
                        if full_text:
                            st.text_area(
                                label=f"Article {i} Text",  # Fixed: Non-empty label
                                value=full_text,
                                height=200,
                                key=f"text_{article['page']}_{article['article_id']}",
                                label_visibility="collapsed"  # Fixed: Hide label but prevent accessibility warning
                            )
                        else:
                            st.info("No text could be extracted from this article.")
            
            # Show all saved crops separately
            if save_crops:
                st.subheader("📸 All Saved Article Crops")
                
                # Try multiple possible crop directories
                from config import OUTPUT_DIR
                possible_dirs = [
                    OUTPUT_DIR / "crops",
                    Path("output/crops"),
                    Path("./output/crops"), 
                    Path("../output/crops"),
                ]
                
                found_crops = False
                for crop_dir in possible_dirs:
                    if crop_dir.exists():
                        crop_images = list(crop_dir.glob("*.jpg")) + list(crop_dir.glob("*.png"))
                        if crop_images:
                            st.success(f"Found {len(crop_images)} crops in {crop_dir}")
                            cols = st.columns(3)
                            for idx, img_path in enumerate(crop_images):
                                with cols[idx % 3]:
                                    st.image(str(img_path), caption=img_path.name, use_container_width=True)
                            found_crops = True
                            break
                
                if not found_crops:
                    st.warning("No crop images found. Check that 'Save cropped images' is enabled and processing completed successfully.")
                    st.info("Crop images should be saved to: output/crops/")
                    
        else:
            st.info("🔍 No education-related articles found.")
            st.info("Try:")
            st.info("• Adjusting the confidence threshold")
            st.info("• Using a different PDF")
            st.info("• Checking if the PDF contains education-related content")

        # Download results
        st.subheader("💾 Download Results")
        json_bytes = json.dumps(results, indent=2, ensure_ascii=False).encode("utf-8")
        filename = f"education_articles_{st.session_state.uploaded_file_name or 'results'}.json"
        st.download_button(
            "📥 Download JSON Results", 
            data=json_bytes, 
            file_name=filename, 
            mime="application/json"
        )
        
        # Reset button
        if st.button("🔄 Process Another PDF", key="reset_btn"):
            # Clear all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]

if __name__ == "__main__":
    main()
