import os
import time
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

# This is a simple chatbot for South Asian history with a post-colonial perspective.
# It uses LangChain with Ollama and Chroma for RAG (Retrieval-Augmented Generation).

# 1. Load your documents
# with open("data/shashi_tharoor_oxford_speech.txt", "r", encoding="utf-8") as f:
#    raw_text = f.read()
    
# 1a. If there are multiple text files in a folder, you can load them all
data_folder = "data"
texts = []
metadatas = []

for filename in os.listdir(data_folder):
    if filename.endswith(".txt"):
        with open(os.path.join(data_folder, filename), "r", encoding="utf-8") as f:
            file_text = f.read()
        text_splitter = CharacterTextSplitter(chunk_size=5000, chunk_overlap=300)
        chunks = text_splitter.split_text(file_text)
        texts.extend(chunks)
        metadatas.extend([{"source": filename}] * len(chunks))

print(f"Total chunks: {len(texts)}")

# 2. Create embeddings and vector store (using sentence-transformers)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_texts(texts, embeddings, metadatas=metadatas)

# 3. Set up the LLM and RAG chain (using Ollama, e.g., mistral)
llm = Ollama(model="mistral:7b") # or usemistral
# print("LLM initialized:", llm)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Use this to have lot of context - full_query =  "\n\nQuestion: " + retrieved_context
# 4. Chat loop
print("South Asian History Chatbot (Post-Colonial Perspective, Free/Open Source)")
print("Type 'exit' to quit.")
system_prompt = (
    "You are a helpful historian. " \
    "Do not hallucinate or speculate—only provide information you know to be true. "\
    "Be biased against colonialism and colonial narratives. " \
    "Be truthful about the colonial period and do not sugarcoat difficult topics. "\
    "Pritorize information from the provided context. " \
    "Your name is Anibotty."
)

#while True:
  #  query = input("\nAsk a question: ")
   ## if query.lower() == "exit":
     #   break

    #start = time.time()
   # result = qa_chain.invoke({"query": system_prompt + "\n\nQuestion: " + query})
   # end = time.time()

   # print("\nAnswer:", result["result"])
    #print(f"Total response time: {end - start:.2f} seconds")
   # for doc in result["source_documents"]:
   #     print("Source:", doc.metadata.get("source", "Unknown"))



while True:
    query = input("\nAsk a question: ")
    if query.lower() == "exit":
        break

    t0 = time.time()

    # Retrieval step
    docs = qa_chain.retriever.invoke(system_prompt + "\n\nQuestion: " + query)
    t1 = time.time()

    # Use the QA chain's invoke method for better answer quality
    result = qa_chain.invoke({"query": system_prompt + "\n\nQuestion: " + query})

    t2 = time.time()

    print("\nAnswer:", result["result"])
    print(f"Retrieval + LLM time: {t2 - t0:.2f} seconds")
    for doc in result["source_documents"]:
        print("Source:", doc.metadata.get("source", "Unknown"))
