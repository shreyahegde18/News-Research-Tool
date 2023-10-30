import os
import streamlit as st
import pickle
import time
import langchain
from langchain.llms import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from secrete import openapi_key
os.environ['OPENAI_API_KEY']= openapi_key
urls=[]
llm=OpenAI(temperature=0.9,max_tokens=500)
file_path="faiss_store_openai.pkl"
st.title("News Research tool 📈")
st.sidebar.title("News Article URLs")
for i in range(3):
    url=st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)
process_url_clicked=st.sidebar.button("Process URLs")

main_placeholder=st.empty()

if process_url_clicked:
    #load data
    loader=UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading... Started...✅✅✅")
    data = loader.load()

    #spli data
    text_splitter=RecursiveCharacterTextSplitter(
        separators=['\n\n','\n','.',','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter... Started...✅✅✅")
    docs=text_splitter.split_documents(data)

    #create embeddings
    embeddings=OpenAIEmbeddings()
    vectorstore_openai=FAISS.from_documents(docs,embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)
    # Save the FAISS index to a pickle file
    with open(file_path, "ab") as f:
        pickle.dump(vectorstore_openai,f)

query= main_placeholder.text_input(("Question: "))
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vector_store=pickle.load(f)
            chain=RetrievalQAWithSourcesChain.from_llm(llm=llm,retriever=vector_store.as_retriever())
            result= chain({"question": query}, return_only_outputs=True)
            # {"answer":"","sources":[]}
            st.header("Answer")
            st.subheader(result["answer"])

            #display sources
            sources=result.get("sources", "")
            if sources:
                st.subheader("Sources: ")
                sources_list=sources.split(("\n"))
                for source in sources_list:
                    st.write(source)