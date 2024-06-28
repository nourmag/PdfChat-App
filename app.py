import streamlit as st
from dotenv import load_dotenv
import pickle
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

# Custom function to add vertical space
def add_vertical_space(lines):
    for _ in range(lines):
        st.write("")

os.environ['OPENAI_API_KEY'] = ''

with st.sidebar:
    st.title('Pdf-Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
    ''')
    add_vertical_space(5)
    st.write('Made by Nour Maged & Naira Mohammed')

def main():
    st.header("Chat with PDF 💬")

    load_dotenv()

    # Uploading PDF file 
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        Pdf_reader = PdfReader(pdf)

        text = ""
        for page in Pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # Embeddings
        store_name = pdf.name[:-4]

        if os.path.exists(f'{store_name}.pkl'):
            with open(f'{store_name}.pkl', 'rb') as f:
                VectorStore = pickle.load(f)
            st.write('Embeddings Loaded from the disk')
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f'{store_name}.pkl', 'wb') as f:
                pickle.dump(VectorStore, f)
            st.write('Embeddings Computation Completed.')

        # Accept user's questions/query
        query = st.text_input('Ask questions about your PDF:')
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)

            llm = OpenAI(model_name="gpt-3.5-turbo-instruct")
            chain = load_qa_chain(llm=llm, chain_type='stuff')
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)

if __name__ == '__main__':
    main()
