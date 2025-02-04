import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load the document
loader = PyPDFLoader("../data/test.pdf")
documents = loader.load()

# Split the document into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
docs = text_splitter.split_documents(documents=documents)

# Load embedding model
embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cuda"}
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs=model_kwargs
)

# Create FAISS vector store
vectorstore = FAISS.from_documents(docs, embeddings)

# Save and reload the vector store
vectorstore.save_local("faiss_index_")
persisted_vectorstore = FAISS.load_local("faiss_index_", embeddings, allow_dangerous_deserialization=True)

# Create a retriever
retriever = persisted_vectorstore.as_retriever()

# Initialize the LLaMA model
llm = OllamaLLM(model="llama3.2")
response = llm.invoke("Tell me a joke")
print(response)

# Create RetrievalQA
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# Interactive query loop
while True:
    query = input("Type your query (or type 'Exit' to quit): \n")
    if query.lower() == "exit":
        break
    result = qa.invoke(query)
    print(result["result"])





