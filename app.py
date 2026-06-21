import streamlit as st
from dotenv import load_dotenv
from utils import fetch_and_process_papers, get_clickable_markdown
from engine import generate_research_plan, update_knowledge_base, load_existing_rag_chain, load_paper_registry

load_dotenv()

st.set_page_config(page_title="Strategic Expert System", layout="wide")
st.title("🎓 Intelligent Research Expert System")

# Initialize global chain in state if it exists locally on file start
if 'rag_chain' not in st.session_state:
    st.session_state['rag_chain'] = load_existing_rag_chain()

# Sidebar inventory overview
with st.sidebar:
    st.header("🧠 System Memory")
    registry = load_paper_registry()
    st.metric("Total Papers Indexed", len(registry))
    if registry:
        with st.expander("Show Indexed Titles"):
            for pid, title in registry.items():
                st.caption(f"• ({pid}) {title}")

topic = st.text_input("Enter a general research topic:")

if st.button("Generate Strategy & Process"):
    if topic:
        with st.status("🏗️ Formulating Structured Research Plan...", expanded=True) as status:
            
            # Step 1: Discover Important Sub-Topics
            st.write("🔍 Mapping comprehensive subtopics via LLM...")
            plan = generate_research_plan(topic)
            
            # Display Plan Metrics dynamically
            st.markdown(f"**Research Summary Plan:** {plan.conceptual_summary}")
            for index, sub in enumerate(plan.subtopics):
                st.markdown(f"👉 **Subtopic {index+1}:** {sub.name}  \n *Why it matters:* {sub.importance}")
            
            # Extract queries from the plan
            queries = [sub.target_search_query for sub in plan.subtopics]
            
            # Step 2: Extract data safely
            st.write("🌐 Cross-referencing queries against local cache and arXiv...")
            new_papers = fetch_and_process_papers(queries)
            
            # Step 3: Incremental Update
            st.write("📥 Updating vector engine incrementally...")
            st.session_state['rag_chain'] = update_knowledge_base(new_papers)
            
            status.update(label="✅ Knowledge Base Updated Successfully!", state="complete")

# Chat Interface
if st.session_state['rag_chain']:
    st.divider()
    query = st.chat_input("Ask anything about your entire dynamic repository...")
    if query:
        with st.spinner("Synthesizing answer..."):
            ans_obj = st.session_state['rag_chain'].invoke(query)
            st.markdown(get_clickable_markdown(ans_obj))
else:
    st.info("Please build or query a topic to initialize the expert agent.")