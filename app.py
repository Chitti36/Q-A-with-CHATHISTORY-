import streamlit as st
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatMessagePromptTemplate,MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import tempfile
from dotenv import load_dotenv
load_dotenv()
os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
embeddings=HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")



## set up streamLIT

st.title("Converstional RAG with PDF uploads and chat History")

st.write("Upload your document and shoot questions")

## Input the Groq API key
api_key=st.text_input("Enter Your groq api key:",type="password")

st.write("If you dont have it use this key as temporary.//  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  //")

## Check  if groq apikey is provided

if api_key:
    llm=ChatGroq(groq_api_key=api_key,model_name="llama3-8b-8192")

   ## chat interface
    session_id=st.text_input("Session ID",value="default session")


    if 'store' not in st.session_state:
        st.session_state.store={}

    uploaded_files = st.file_uploader("Go choose your file!", type="pdf", accept_multiple_files=True)


    ## process uploadd PDFs

    if uploaded_files:
        documents=[]

        for uploaded_file in uploaded_files:
            temppdf=f"./temp.pdf"
           with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
               
                tmpfile.write(uploaded_file.getvalue())
                tmpfile.flush()  # ensure file is fully written to disk
    
                loader = PyPDFLoader(tmpfile.name)
                docs = loader.load()
                documents.extend(docs)

    #split and create embedding for the documents

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits=text_splitter.split_documents(documents)
        vectorstore=FAISS.from_documents(splits,embeddings)
        retriever=vectorstore.as_retriever()


        contextualize_a_system_prompt=(
            "Given a chat history and latest user question"
            "which might referenece context in the chat history"
            "formulate a standalone question which can be understood"
            "without the chat history.do NOT answer the question"
            " just reformualte it if need otherwise return it as it is "
        )


        contextualize_a_prompt=ChatPromptTemplate.from_messages(
            [
                ("system",contextualize_a_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}")
            ]
        )

        history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_a_prompt)


        ## Answer question
        system_prompt=(
            "You are an assistant for question-answering"
            "Use following pieces of retrieved context to answer "
            "the question.If you dont know just say you dont "
            "know.Use three sentences at max and keep the answer concise"
            "\n\n"
            "{context}"
        )

        qa_prompt=ChatPromptTemplate.from_messages(
            [
                ("system",system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}"),
            ]
        )
        question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
        rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)


        def getsessionhistory(session:str)->BaseChatMessageHistory:
            if  session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
            

        conversation_ragchain=RunnableWithMessageHistory(
            rag_chain,getsessionhistory,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"

        )

        user_input=st.text_input("your question")
        if user_input:
            session_history=getsessionhistory(session_id)
            response=conversation_ragchain.invoke(
                {"input": user_input},
                config={
                    "configurable":{"session_id":session_id}

                },
            )
            st.write(st.session_state.store)
            st.write("assistant:",response['answer'])
            st.write("Chat History:",session_history.messages)


## REF:https://chatgpt.com/c/685c5745-5790-800a-b670-4e3bac334435






