import logging
import streamlit as st
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

try:
    from rag.engine import ask_grid_vision
except ImportError as e:
    logger.critical(f"Failed to import RAG engine: {e}")
    st.error("Critical System Error: Failed to load RAG Engine. Check server logs.")
    st.stop()

st.set_page_config(
    page_title="GridVision Assistant",
    layout="wide"
)

st.markdown("""
<style>
    .source-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        border-left: 5px solid #ff4b4b;
    }
</style>
""", unsafe_allow_html=True)

def display_sources(sources: List[Dict[str, Any]]) -> None:
    with st.expander("Found Sources (Click to expand)", expanded=True):
        cols = st.columns(3)
        sorted_sources = sorted(sources, key=lambda x: x.get('type') != 'image')

        for i, source in enumerate(sorted_sources):
            with cols[i % 3]:
                st.markdown(f"<div class='source-card'>", unsafe_allow_html=True)
                if source.get('type') == 'image' and source.get('image_path'):
                    try:
                        img_rel_path = Path(source['image_path'])
                        full_img_path = (BASE_DIR / img_rel_path).resolve()
                        
                        if not str(full_img_path).startswith(str(BASE_DIR)):
                            logger.warning(f"Security Alert: Path traversal attempt detected: {source['image_path']}")
                            st.warning("Security Warning: Invalid image path detected.")
                        elif full_img_path.exists():
                            st.image(str(full_img_path), caption=f"Schematic: {source.get('source', 'Doc')}", use_column_width=True)
                        else:
                            logger.warning(f"Image file missing: {full_img_path}")
                            st.warning(f"File not found: {source['image_path']}")
                            
                    except Exception as e:
                        logger.error(f"Error processing image path: {e}")
                        st.warning("Error loading image.")
                else:
                    st.markdown(f"**Document:** {source.get('source', 'N/A')}")
                    st.caption(f"Page: {source.get('page', 'N/A')} | Score: {source.get('score', 0):.2f}")
                    content_preview = source.get('content', '')[:150]
                    st.text(f"{content_preview}...")
                
                st.markdown("</div>", unsafe_allow_html=True)

def main() -> None:
    with st.sidebar:
        st.title("GridVision RAG")
        st.markdown("---")
        st.info(
            """
            **Technical Support System**
            
            The assistant searches for answers in:
            - Text Manuals
            - Wiring Schematics
            - Diagnostic Tables
            """
        )
        if st.button("Clear History"):
            st.session_state.messages = []
            st.rerun()

    st.title("Technical Assistant")
    st.caption("Enter a query, e.g., 'Show me the wiring diagram for K1 relay' or 'How to test the fuel pump?'")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                display_sources(message["sources"])

    if prompt := st.chat_input("Your question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing schematics and manuals..."):
                try:
                    logger.info(f"Processing query: {prompt}")
                    response_text, sources = ask_grid_vision(prompt)
                    
                    st.markdown(response_text)
                    if sources:
                        display_sources(sources)
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response_text,
                        "sources": sources
                    })
                    
                except Exception as e:
                    logger.error(f"Error responding to user: {e}", exc_info=True)
                    st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()