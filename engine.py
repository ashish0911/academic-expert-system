import os
import json
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from models import ExpertAnswer, TopicPlan

DB_PATH = "faiss_research_index"
REGISTRY_PATH = "processed_papers_registry.json"

def generate_research_plan(topic: str) -> TopicPlan:
    """Uses LLM to break down a topic into structured subtopics and queries."""
    planner_llm = ChatOpenAI(model="gpt-4o", temperature=0.2).with_structured_output(TopicPlan)
    prompt = ChatPromptTemplate.from_template(
        "You are a principal research scientist. Break down the following topic into its core subtopics, "
        "identifying what needs to be investigated, and create optimized search queries for each: {topic}"
    )
    chain = prompt | planner_llm
    return chain.invoke({"topic": topic})

def load_paper_registry():
    """Loads a ledger of papers we have already vectorized to prevent double-spending."""
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_paper_registry(registry):
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=4)

def update_knowledge_base(new_paper_data):
    """Incrementally updates the FAISS database and registry with new documents."""
    if not new_paper_data:
        return load_existing_rag_chain()
        
    embeddings = OpenAIEmbeddings()
    
    # Process text files into chunks
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    registry = load_paper_registry()
    
    for p in new_paper_data:
        chunks = splitter.split_text(p['text'])
        for c in chunks:
            docs.append(Document(page_content=c, metadata={"source": p['title'], "url": p['url'], "id": p['id']}))
        # Record the paper into our persistent ledger
        registry[p['id']] = p['title']
        
    save_paper_registry(registry)

    # Incremental Vector Storage Setup
    if os.path.exists(DB_PATH):
        # Load old index and add new documents seamlessly
        vector_db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
        vector_db.add_documents(docs)
        vector_db.save_local(DB_PATH)
    else:
        # Build index for the first time
        vector_db = FAISS.from_documents(docs, embeddings)
        vector_db.save_local(DB_PATH)
        
    return build_rag_chain(vector_db)

def load_existing_rag_chain():
    """Loads the database from disk directly if no new papers were added."""
    if os.path.exists(DB_PATH):
        vector_db = FAISS.load_local(DB_PATH, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
        return build_rag_chain(vector_db)
    return None

def build_rag_chain(vector_db):
    """Constructs the standard LCEL RAG execution flow."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    doc_prompt = PromptTemplate.from_template("TITLE: {source}\nURL: {url}\nID: {id}\nCONTENT: {page_content}\n---")
    
    def format_docs(docs):
        return "\n\n".join([doc_prompt.format(**d.metadata, page_content=d.page_content) for d in docs])

    expert_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a research expert. Answer using context and numeric citations [1]."),
        ("human", "CONTEXT:\n{context}\n\nQUESTION: {input}"),
    ])

    return (
        {"context": vector_db.as_retriever(search_kwargs={"k": 5}) | format_docs, "input": RunnablePassthrough()}
        | expert_prompt
        | llm.with_structured_output(ExpertAnswer)
    )