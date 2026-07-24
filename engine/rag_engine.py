import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

class RAGEngine:
    def __init__(self, data_dir: str = ".", api_key: str = None):
        self.data_dir = data_dir
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.embeddings = OpenAIEmbeddings(api_key=self.api_key)
        self.vectorstore = None
        self.llm = ChatOpenAI(model="gpt-4o-mini", api_key=self.api_key, temperature=0)
        self.store = {}

    def ingest_documents(self):
        print(f"Loading all PDFs from directory: {self.data_dir}")
        loader = PyPDFDirectoryLoader(self.data_dir)
        docs = loader.load()
        
        if not docs:
            print("No PDFs found to ingest.")
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_docs = text_splitter.split_documents(docs)
        
        print(f"Processing documents into Vector DB...")
        self.vectorstore = Chroma.from_documents(documents=all_docs, embedding=self.embeddings)
        print("Vector DB ready!")

    def get_conversational_chain(self):
        if not self.vectorstore:
            self.ingest_documents()
            
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        
        system_prompt = (
            "You are an advanced AI Assistant and Enterprise Knowledge Agent.\n"
            "Use the given context to answer the questions accurately.\n"
            "If you don't know the answer, say that you don't know.\n\n"
            "{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}")  # <-- Yahan 'input' hona zaroori hai history chain ke liye
        ])
        
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in self.store:
                self.store[session_id] = InMemoryChatMessageHistory()
            return self.store[session_id]
            
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
        
        return conversational_rag_chain