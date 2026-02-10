import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain_pinecone import PineconeVectorStore

#For RAG example we import more modules 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter


print('Initializing components....')

llm = ChatOpenAI(temperature=0)
embeddings = OpenAIEmbeddings()

vectorstore=PineconeVectorStore(index_name=os.getenv("INDEX_NAME"), embedding=embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Selecting the top 3 most relevant chunks from the vector store


prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:
    {context}

    Question:{question}

    Provide a detailed answer

    """
)

def format_docs(docs):
    """Format retrieved documents into a single string"""

    return '\n\n'.join(doc.page_content for doc in docs)


def retrieval_chain_without_lcel(query:str):
    """A simple retrival chain that retrieves relevant documents and formats them into a prompt for the LLM"""

    # Step 1: Retrieve relevant documents from the vector store
    retrieved_docs = retriever.invoke(query)

    # Step 2: Format the retrieved documents into a single string
    context = format_docs(retrieved_docs)

    # Step 3: Create the prompt by filling in the template with the context and question
    prompt = prompt_template.format_messages(context=context, question=query)

    # Step 4: Invoke the LLM with the formatted prompt
    response = llm.invoke(prompt)

    return response.content


# Method for RAG with LCE (Language Chain Execution Language)
def create_retrieval_chain_with_lcel():
    """A retrieval chain that uses LCE to execute the retrieval and formatting steps"""

    retrieval_chain =(
        RunnablePassthrough.assign(
            context = itemgetter("question") | retriever | format_docs
        ) | prompt_template | llm | StrOutputParser()
    )

    return retrieval_chain



if __name__ =='__main__':


    query = "What is Pinecone in AI/Machine learning?"
    # Raw invocation without RAG
    print("\n" + "="*70)
    print("Implementing RAG with a Naive Retrieval Approach (No RAG)")
    print("="*70)
    result_raw = llm.invoke([HumanMessage(content=query)])
    print("\nAnswer")
    print(result_raw.content)

    # RAG implementation with a naive retrieval approach
    print("\n" + "="*70)
    print("Implementing RAG with a Naive Retrieval Approach (RAG)")
    print("="*70)
    result_rag = retrieval_chain_without_lcel(query)
    print("\nAnswer")
    print(result_rag)


    # RAG implementation with LCE
    print("\n" + "="*70)
    print("Implementing RAG with LCE")
    print("="*70)
    retrieval_chain_with_lcel = create_retrieval_chain_with_lcel()
    result_lcel = retrieval_chain_with_lcel.invoke({"question": query})
    print("\nAnswer")
    print(result_lcel)