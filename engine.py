import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_core.runnables import RunnableLambda
from langchain_core.embeddings import Embeddings

from models import ExpertAnswer, TopicPlan
from utils import load_paper_registry, save_paper_registry, REGISTRY_PATH

QDRANT_HOST = os.getenv("QDRANT_HOST", "http://localhost:6333")
COLLECTION_NAME = "research_papers"

class TrackedEmbeddings(Embeddings):
    """Wraps embedding generation to log costs while passing strict type validation."""
    def __init__(self, base_embeddings, cb_handler=None):
        self.base_embeddings = base_embeddings
        self.cb_handler = cb_handler

    def embed_documents(self, texts):
        if self.cb_handler:
            # Route chunk parsing through an isolated, traceable LangChain runnable
            runnable = RunnableLambda(self.base_embeddings.embed_documents)
            return runnable.invoke(texts, config={
                "callbacks": [self.cb_handler],
                "run_name": "OpenAI-Embedding-Generation"
            })
        return self.base_embeddings.embed_documents(texts)

    def embed_query(self, text):
        return self.base_embeddings.embed_query(text)


def get_vector_store(cb_handler=None):
    """Initializes Qdrant safely using the clean type-validated telemetry wrapper."""
    client = QdrantClient(url=QDRANT_HOST)
    base_embeddings = OpenAIEmbeddings()
    
    embeddings = TrackedEmbeddings(base_embeddings, cb_handler)
    
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )
    return QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embeddings)

def generate_research_plan(topic: str, cb_handler=None) -> TopicPlan:
    planner_llm = ChatOpenAI(model="gpt-4o", temperature=0.2).with_structured_output(TopicPlan)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a principal research scientist. Break down the given topic into its core critical subtopics, explain their importance, and formulate optimized search queries for arXiv."),
        ("human", "Topic to investigate: {topic}")
    ])
    chain = prompt | planner_llm
    
    config = {}
    if cb_handler:
        config["callbacks"] = [cb_handler]
    return chain.invoke({"topic": topic}, config=config)

def update_knowledge_base(new_paper_data, cb_handler=None):
    vector_store = get_vector_store(cb_handler=cb_handler)
    if not new_paper_data:
        return build_production_rag_chain(vector_store)
        
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    
    for p in new_paper_data:
        chunks = splitter.split_text(p['text'])
        for c in chunks:
            docs.append(Document(page_content=c, metadata={"source": p['title'], "url": p['url'], "id": p['id']}))
        
    vector_store.add_documents(docs)
    
    registry = load_paper_registry()
    for p in new_paper_data:
        registry[p['id']] = p['title']
    save_paper_registry(registry)
    
    return build_production_rag_chain(vector_store)

def clear_all_data():
    """Purges the local running Qdrant storage metrics and cache ledger files completely."""
    client = QdrantClient(url=QDRANT_HOST)
    if client.collection_exists(collection_name=COLLECTION_NAME):
        client.delete_collection(collection_name=COLLECTION_NAME)
        
    if os.path.exists(REGISTRY_PATH):
        os.remove(REGISTRY_PATH)

def build_production_rag_chain(vector_store=None):
    if vector_store is None:
        vector_store = get_vector_store()
        
    client = QdrantClient(url=QDRANT_HOST)
    if not client.collection_exists(collection_name=COLLECTION_NAME) or client.get_collection(collection_name=COLLECTION_NAME).points_count == 0:
        return None
        
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 15})
    
    from flashrank import Ranker
    import shutil
    from pathlib import Path
    
    model_name = "ms-marco-MultiBERT-L-12"
    cache_directory = "./flashrank_models"  # Persistent directory in your workspace
    
    try:
        ranker_client = Ranker(model_name=model_name, cache_dir=cache_directory)
    except Exception:
        corrupted_path = Path(cache_directory) / model_name
        if corrupted_path.exists():
            shutil.rmtree(corrupted_path)
            
        ranker_client = Ranker(model_name=model_name, cache_dir=cache_directory)

    compressor = FlashrankRerank(client=ranker_client, top_n=4)
    
    rerank_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)
    
    doc_prompt = PromptTemplate.from_template("TITLE: {source}\nURL: {url}\nID: {id}\nCONTENT: {page_content}\n---")
    def format_docs(docs):
        return "\n\n".join([doc_prompt.format(**d.metadata, page_content=d.page_content) for d in docs])

    expert_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a top-tier research expert. Answer the user's question completely based ONLY on the provided context pieces. Use numeric citations like [1] or [1, 2] in your text when referencing information."),
        ("human", "CONTEXT:\n{context}\n\nQUESTION: {input}"),
    ])

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    return ({"context": rerank_retriever | format_docs, "input": RunnablePassthrough()} | expert_prompt | llm.with_structured_output(ExpertAnswer))