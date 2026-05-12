import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from models import SearchRequest
from utils import fetch_and_process_papers, get_clickable_markdown
from engine import setup_expert_system

load_dotenv() # Automatically looks for .env

st.set_page_config(page_title="Expert Research System", layout="wide")
st.title("🎓 Academic Expert System")

topic = st.text_input("Enter a research topic:")

if st.button("Build Knowledge Base"):
    if not topic:
        st.warning("Please enter a topic first.")
    else:
        # Use st.status to show a running log of background activities
        with st.status("🚀 Initializing Expert Brain...", expanded=True) as status:
            
            # Step 1: Query Generation
            st.write("🤖 LLM is brainstorming specialized search queries...")
            query_llm = ChatOpenAI(model="gpt-4o").with_structured_output(SearchRequest)
            query_resp = query_llm.invoke(f"Generate 3 arXiv search queries for: {topic}")
            st.write(f"✅ Generated Queries: {', '.join(query_resp.queries)}")
            
            # Step 2: Fetching and Processing
            st.write("🌐 Connecting to arXiv and downloading PDFs...")
            # We add a progress bar specifically for the papers
            papers = fetch_and_process_papers(query_resp.queries)
            st.write(f"📚 Successfully retrieved {len(papers)} research papers.")
            
            # Step 3: RAG Setup
            st.write("🧠 Vectorizing text and building index (this takes a moment)...")
            st.session_state['rag_chain'] = setup_expert_system(papers)
            
            # Finalize status
            status.update(label="✅ Knowledge Base Complete!", state="complete", expanded=False)
        
        st.success(f"Expert System is now trained on {len(papers)} papers!")

if 'rag_chain' in st.session_state:
    query = st.chat_input("Ask a question about the papers...")
    if query:
        with st.spinner("Thinking..."):
            ans_obj = st.session_state['rag_chain'].invoke(query)
            st.markdown(get_clickable_markdown(ans_obj))