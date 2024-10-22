# History_of_Nepal.py

import os
import textwrap
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Qdrant
from langchain_groq import ChatGroq
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.prompts import ChatPromptTemplate
from llama_parse import LlamaParse
from langchain_community.vectorstores import Qdrant
from pydantic import BaseModel



# Set the API keys and paths
os.environ["GROQ_API_KEY"] = "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"
llama_api_key = 'llx-Axr7AFeJbw05sqrAwtG7nVjKxFohz4kN9oCd9CQoIT400C0o'
utf8_document_path = r'D:\Downloads\IDM\Parsed_Documents\parsed_documents_utf8.md'

# Load and parse documents
def load_documents():
    loader = UnstructuredMarkdownLoader(utf8_document_path)
    loaded_documents = loader.load()
    return loaded_documents

# Set up the embeddings and retriever
def setup_retriever(loaded_documents):
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    qdrant = Qdrant.from_documents(
        loaded_documents,
        embeddings,
        path=r'D:\Downloads\project\qdrant_db',
        collection_name='History0fNepal_embeddings'
    )
    return qdrant.as_retriever(search_kwargs={"k": 5})
# def setup_retriever(loaded_documents):
#     embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")
#     qdrant = Qdrant.from_documents(
#         loaded_documents,
#         embeddings,
#         collection_name='HistoryOfNepal_embeddings',
#         url="http://localhost:6333"  # Assuming Qdrant server is running on default port
#     )
#     return qdrant.as_retriever(search_kwargs={"k": 5})


# Initialize the LLM
def setup_llm():
    llm = ChatGroq(temperature=0.01, model='mixtral-8x7b-32768')
    return llm

# Create the QA chain
def create_qa_chain(retriever):
    prompt_template = """
    Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}

    Answer the question and provide additional helpful information, based on the pieces of information, if applicable. Be succinct.
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    
    qa = RetrievalQA.from_chain_type(
        llm=setup_llm(),
        chain_type='stuff',
        retriever=retriever,
        chain_type_kwargs={'prompt': prompt, "verbose": False}
    )
    return qa

# Get answer based on the question
def get_answer(question):
    loaded_documents = load_documents()
    retriever = setup_retriever(loaded_documents)
    qa_chain = create_qa_chain(retriever)
    
    # Invoke the QA chain
    response = qa_chain.invoke(question)
    
    # Format and return the response
    return response["result"]

if __name__ == "__main__":
    # Run a test question if you want to debug the script
    print(get_answer("Where is Nepal?"))

# class MyModel(BaseModel):
#     class Config:
#         arbitrary_types_allowed = True