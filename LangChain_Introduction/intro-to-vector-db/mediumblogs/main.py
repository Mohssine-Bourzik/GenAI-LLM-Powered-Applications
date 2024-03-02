import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_pinecone import Pinecone
from langchain.chains import RetrievalQA
import pinecone
from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    print("Hello VectorStore!")
    loader = TextLoader(
        "/path_to_the_txt_file/mediumblog1.txt"
    )
    document = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(len(texts))

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    docsearch = Pinecone.from_documents(
        texts, embeddings, index_name="your_pinecone_index_name"
    )

    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type= "stuff",retriever=docsearch.as_retriever(),
    )

    query = "What is a vector DB? Give me a 15 word answer for a beginner"
    result = qa.run({"query": query}) 
    print(result)