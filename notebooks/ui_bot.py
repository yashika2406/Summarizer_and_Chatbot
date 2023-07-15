import openai
import os
from tika import parser
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Milvus
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
# from transformers import pipelines
# from dotenv import load_dotenv
from apikey import API_KEY
from langchain.chains import LLMChain, ConversationChain
from langchain.callbacks import get_openai_callback
from langchain.chains.conversation.memory import (ConversationBufferMemory)
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
import streamlit as st

MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"

os.environ['OPENAI_API_KEY'] = API_KEY;


def main():
    # load_dotenv()
    st.set_page_config(page_title="Enter your file : ")
    st.header("ENTER YOUR FILE")
    file = st.file_uploader("Upload your file")
    parsed_file = parser.from_file(file)
    data = parsed_file['content']
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100
    )

    # creating chunks and cleaning those chunks
    chunks = splitter.split_text(text=data)

    # print(chunks)

    def clean_text(text):
        cleaned_string = text.replace("\n", "").replace('..', "")
        return cleaned_string

    clean_chunks = [clean_text(para) for para in chunks]
    # st.write(clean_chunks)

    embedding = OpenAIEmbeddings()
    embed_chunk_base = Milvus.from_texts(clean_chunks, embedding,
                                         connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT}, drop_old=True)

    query = st.text_input("Enter your query : ")
    if query:
        llm = OpenAI(temperature=0.5)
        filter_chunk = embed_chunk_base.similarity_search(query)
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=filter_chunk, question=query)
        st.write(response)


if __name__ == "__main__":
    main()
