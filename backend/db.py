# import chromadb
from sentence_transformers import SentenceTransformer
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

class KnowledgeBase:
    def __init__(self, engine="all-MiniLM-L6-v2"):
        self.engine = engine

    def get_text_from_url(self, url):
        reqs = requests.get(url)
        soup = BeautifulSoup(reqs.content, "html.parser")
        content = soup.find('div', {'class': 'mw-body-content'})
        return content.get_text(separator="\n", strip=True)
    
    def persist_embeddings_in_chroma(self, url="https://en.wikipedia.org/wiki/Stock_market"):
        # Logic to generate embeddings using the model
        text = self.get_text_from_url(url)
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        docs = splitter.create_documents([text])
        
        embeddings = SentenceTransformerEmbeddings(model_name=self.engine)
        
        chroma = Chroma.from_documents(docs, embeddings, persist_directory="finance_101")
        chroma.persist()

        return True