import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

print('Loading the docs...')
docs = TextLoader("C:/Users/choll/Desktop/Studeis/My studies/LangChain with Langraph Basic LLM/langchain-course/mediumblog1.txt",encoding='utf-8')
print('Got the documents, next splitting the text into chunks...')

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
chunks = text_splitter.split_documents(docs.load())
print(f'Text splitted into chunks, created {len(chunks)}, next creating the embeddings and ingesting into Pinecone...')

embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
vectorstore = PineconeVectorStore.from_documents(chunks,embeddings,index_name=os.getenv("INDEX_NAME"))
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


print(f'Components initialized, next creating the prompt template and the retrieval chain...')

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:
    {context}

    Question:{question}

    Provide a detailed answer

    """
)

llm = ChatOpenAI(temperature=0)

def format_docs(docs):
    """Format retrieved documents into a single string"""

    return '\n\n'.join(doc.page_content for doc in docs)

def create_retrieval_chain_with_lcel(query:str):
    """A simple retrival chain that retrieves relevant documents and formats them into a prompt for the LLM, using LCEL to format the retrieved documents"""

    retrieval_chain = (
        {
            'context':retriever| format_docs,
            'question':RunnablePassthrough()
        }| prompt_template|llm|StrOutputParser()
    )

    return retrieval_chain.invoke(query)

if __name__ == '__main__':
    response = create_retrieval_chain_with_lcel("What is the main topic of the blog post?")
    print("Response from the retrieval chain with LCEL:")
    print(response)

