import streamlit as st
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from rag.engine import ask_grid_vision

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

def main():
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
                    st.error(f"An error occurred: {e}")

def display_sources(sources):
    with st.expander("Found Sources (Click to expand)", expanded=True):
        cols = st.columns(3)
        
        sorted_sources = sorted(sources, key=lambda x: x.get('type') != 'image')

        for i, source in enumerate(sorted_sources):
            with cols[i % 3]:
                st.markdown(f"<div class='source-card'>", unsafe_allow_html=True)
                
                if source.get('type') == 'image' and source.get('image_path'):
                    full_img_path = os.path.join(BASE_DIR, source['image_path'])
                    if os.path.exists(full_img_path):
                        st.image(full_img_path, caption=f"Schematic: {source.get('source', 'Doc')}", use_column_width=True)
                    else:
                        st.warning(f"File not found: {source['image_path']}")
                else:
                    st.markdown(f"**Document:** {source.get('source', 'N/A')}")
                    st.caption(f"Page: {source.get('page', 'N/A')} | Score: {source.get('score', 0):.2f}")
                    st.text(f"{source.get('content', '')[:150]}...")
                
                st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()