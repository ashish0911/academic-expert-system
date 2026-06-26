import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

from langfuse import observe, get_client
from langfuse.langchain import CallbackHandler
from utils import fetch_and_process_papers, get_clickable_markdown, load_paper_registry
from engine import generate_research_plan, update_knowledge_base, build_production_rag_chain, clear_all_data

st.set_page_config(page_title="Strategic Enterprise Expert System", layout="wide")
st.title("🎓 Production Research Expert System")

if 'rag_chain' not in st.session_state:
    try:
        st.session_state['rag_chain'] = build_production_rag_chain()
    except Exception:
        st.session_state['rag_chain'] = None

@observe(name="Knowledge-Base-Ingestion-Pipeline")
def run_knowledge_base_build(topic_str):
    langfuse_handler = CallbackHandler()
    
    st.write("🔍 Mapping comprehensive domain subtopics using the Planner Agent...")
    plan = generate_research_plan(topic_str, cb_handler=langfuse_handler)
    
    st.markdown(f"### Research Strategy Plan for: *{plan.main_topic}*")
    st.markdown(f"**Conceptual Framework:** {plan.conceptual_summary}")
    for index, sub in enumerate(plan.subtopics):
        st.markdown(f"👉 **Subtopic {index+1}:** **{sub.name}** \n*Context:* {sub.importance}")
    
    queries = [sub.target_search_query for sub in plan.subtopics]
    
    st.write("🌐 Cross-referencing query strings with local cache tracker...")
    new_papers = fetch_and_process_papers(queries)
    
    st.write("📥 Embedding and loading text fragments to cloud-native Qdrant server...")
    st.session_state['rag_chain'] = update_knowledge_base(new_papers, cb_handler=langfuse_handler)
    
    get_client().flush()

@observe(name="Research-QA-Query")
def run_chatbot_query(user_query):
    langfuse_handler = CallbackHandler()
    ans_obj = st.session_state['rag_chain'].invoke(user_query, config={"callbacks": [langfuse_handler]})
    get_client().flush()
    st.markdown(get_clickable_markdown(ans_obj))

with st.sidebar:
    st.header("🧠 Persistent Vector Storage")
    registry = load_paper_registry()
    st.metric("Total Unique Papers Ingested", len(registry))
    if registry:
        with st.expander("Ingested Repository Inventory"):
            for pid, title in registry.items():
                st.caption(f"• ({pid}) {title}")
                
    st.divider()
    st.subheader("🛠️ Database Administration")
    if st.button("🗑️ Wipe Qdrant Collection & Cache", type="primary", use_container_width=True):
        with st.spinner("Purging storage indices..."):
            clear_all_data()
            st.session_state['rag_chain'] = None
        st.success("Database and local JSON cache cleared successfully!")
        st.rerun()

topic = st.text_input("Enter a general research topic area to construct/augment storage:")

if st.button("Generate Strategy & Process"):
    if not topic:
        st.warning("Please type a topic domain first.")
    else:
        with st.status("🏗️ Executing Strategic Research Pipeline...", expanded=True) as status:
            run_knowledge_base_build(topic)
            status.update(label="✅ Knowledge Base Update Complete!", state="complete")
        st.success("The system cache has updated successfully. Ask a question below!")

if st.session_state.get('rag_chain') is not None:
    st.divider()
    query = st.chat_input("Ask a question spanning the combined literature collection...")
    if query:
        with st.spinner("Synthesizing answer using cross-document FlashRank re-ranking..."):
            run_chatbot_query(query)
else:
    st.info("The knowledge database is currently empty. Run a topic search query above to initiate the expert agent.")