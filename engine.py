from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from models import ExpertAnswer

def setup_expert_system(paper_data):
    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
    
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    for p in paper_data:
        chunks = splitter.split_text(p['text'])
        for c in chunks:
            docs.append(Document(page_content=c, metadata={"source": p['title'], "url": p['url'], "id": p['id']}))
    
    vector_db = FAISS.from_documents(docs, embeddings)
    doc_prompt = PromptTemplate.from_template("TITLE: {source}\nURL: {url}\nID: {id}\nCONTENT: {page_content}\n---")
    
    def format_docs(docs):
        return "\n\n".join([doc_prompt.format(**d.metadata, page_content=d.page_content) for d in docs])

    expert_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a research expert. Answer the user's question based ONLY on the provided context. "
            "Use simple numeric citations like [1], [2] in your text. "
            "Only cite a paper if it directly supports your statement. "
            "Return the answer and a structured list of sources used."
        )),
        ("human", "CONTEXT:\n{context}\n\nQUESTION: {input}"),
    ])

    chain = (
        {"context": vector_db.as_retriever(search_kwargs={"k": 5}) | format_docs, "input": RunnablePassthrough()}
        | expert_prompt
        | llm.with_structured_output(ExpertAnswer)
    )
    return chain