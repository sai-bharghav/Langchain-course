import os 
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore



load_dotenv()

if __name__ == '__main__':
    print('Ingesting')
    loader = TextLoader("C:/Users/choll/Desktop/Studeis/My studies/LangChain with Langraph Basic LLM/langchain-course/mediumblog1.txt",encoding='utf-8')
    try:
        documents = loader.load()
        print(f"Loaded {len(documents)} documents.")
    except Exception as e:
        print(f"Could load the documents, Error loading documents: {e}")
    
    print("Starting text splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    try:
        chunks = text_splitter.split_documents(documents)
        print(f'Created {len(chunks)} chunks.')
    except Exception as e:
        print(f"Error during text splitting: {e}")

    print('Chunks created:')

    print('Starting embedding generation...')
    
    try:
        embeddings = OpenAIEmbeddings(model='text-embedding-3-small', chunk_size=1000)
        print("Embeddings object created successfully.")
    except Exception as e:
        print(f"Error creating embeddings object: {e}")

    print('Ingesting the embeddings into Pinecone...')

    try: 
        PineconeVectorStore.from_documents(chunks, embeddings, index_name=os.getenv("INDEX_NAME"))
        print("Embeddings ingested into Pinecone successfully.")
    except Exception as e:
        print(f"Error ingesting embeddings into Pinecone: {e}")
