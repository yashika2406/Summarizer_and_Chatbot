import openai
import os
from tika import parser
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Milvus
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
# from transformers import pipelines
from apikey import API_KEY
from langchain.chains import LLMChain, ConversationChain
from langchain.callbacks import get_openai_callback
from langchain.chains.conversation.memory import (ConversationBufferMemory)
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import LLMChain
import logging


MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"

os.environ['OPENAI_API_KEY'] = API_KEY;

def create_log():
    logging.basicConfig(filename='logger.log',format='%(asctime)s %(message)s', filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)



def main(path):
    # load_dotenv()
    create_log()
    #Taking input file from the path that user provide
    parsed_file = parser.from_file(path)
    data = parsed_file['content']
    splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    def clean_text(text):
        cleaned_string = text.replace("\n","").replace('..',"")
        return cleaned_string

    #cleaning of data
    cleaned_data = clean_text(data)

    #creating chunks
    clean_chunks = splitter.split_text(text=data)

    # print(len(clean_chunks))

    #Summarization and chatbot
    choice = input("Do you want summarization or not: ")

    if choice[0] == 'y' or choice[0] == 'Y':
        words = input("How many words do you want? ")
        if not words :
            words = '100'
        summary(cleaned_data,int(words))
    print("CHAT BOT STARTED : ")
    chatbot_ai(clean_chunks)


#function for summarization of document
def summary(data,word):
    prompt = PromptTemplate(
        input_variables=["words", "data"],
        template="Summarize the below document in maximum {words} words: {data}",
    )
    llm = OpenAI(temperature=0.3)
    chain = LLMChain(llm=llm, prompt=prompt)
    print(chain.run({'words' : word,"data" : data}))
    # summarizer = pipeline('summarization')
    # summary = summarizer(data, max_length=50, min_length = 10)
    # print(summary)



#function for chatbot
def chatbot_ai(clean_chunks):

    # llm = OpenAI(temperature=0.5)
    # print(len(clean_chunks))
    #create the embedding of the datamodel
    embedding = OpenAIEmbeddings()
    embed_chunk_base = Milvus.from_texts(clean_chunks,embedding,connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT}, drop_old=True)
    print(embed_chunk_base)

    #Storing conversation in memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_bot = ConversationalRetrievalChain.from_llm(OpenAI(model_name="gpt-3.5-turbo",temperature=0.5) , embed_chunk_base.as_retriever(), memory=memory, verbose=True)


    #Take the input query and show ai response
    while True :
        query = input("Please enter you query : ")
        result = qa_bot({'question': query})
        print(result['answer'])

        again = input("Do you wanna ask anything? : ")
        if again[0]!="Y" and again[0]!="y" :
            print(result['chat_history'])
            break
#if user don't want to continue chat then it will show the chat history and session will end.

#user will enter the file path
if __name__ == '__main__':
    path = input("Enter file path : ")
    main(path)
