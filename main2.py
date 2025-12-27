from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Load environment variables
load_dotenv()

# =============================================================================
# 1. RAG (Retrieval-Augmented Generation) System
# =============================================================================
def create_rag_system():
    """
    RAG allows the LLM to answer questions based on your own documents.
    It retrieves relevant information and uses it to generate answers.
    """
    print("\n=== RAG System Demo ===\n")
    
    # Sample documents (in real use, load from files/database)
    documents = [
        Document(page_content="LangChain is a framework for developing applications powered by language models."),
        Document(page_content="RAG stands for Retrieval-Augmented Generation. It combines retrieval and generation."),
        Document(page_content="Vector databases store embeddings which are numerical representations of text."),
        Document(page_content="FAISS is a library for efficient similarity search developed by Facebook AI."),
        Document(page_content="Embeddings capture semantic meaning, so similar texts have similar embeddings."),
    ]
    
    # Split documents into chunks (important for long documents)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20
    )
    splits = text_splitter.split_documents(documents)
    
    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    # Create RAG prompt template
    template = """Answer the question based only on the following context:

Context: {context}

Question: {question}

Answer in a clear and concise way:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Create the RAG chain
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Test the RAG system
    question = "What is RAG and how does it work?"
    answer = rag_chain.invoke(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}\n")
    
    return rag_chain

# =============================================================================
# 2. Conversational Chain with Memory
# =============================================================================
def create_conversational_chain():
    """
    This chain remembers previous messages in the conversation,
    allowing for multi-turn dialogues with context.
    """
    print("\n=== Conversational Chain with Memory Demo ===\n")
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    
    # Create prompt with memory placeholder
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Keep your responses concise."),
        MessagesPlaceholder(variable_name="history"),
        ("user", "{input}")
    ])
    
    # Initialize memory
    memory = ConversationBufferMemory(return_messages=True, memory_key="history")
    
    # Create chain
    chain = prompt | llm | StrOutputParser()
    
    # Simulate a conversation
    conversations = [
        "My name is Alice and I love Python programming.",
        "What's my name?",
        "What programming language do I like?"
    ]
    
    for user_input in conversations:
        # Get chat history
        history = memory.load_memory_variables({})["history"]
        
        # Get response
        response = chain.invoke({"input": user_input, "history": history})
        
        # Save to memory
        memory.save_context({"input": user_input}, {"output": response})
        
        print(f"User: {user_input}")
        print(f"Assistant: {response}\n")
    
    return chain, memory

# =============================================================================
# 3. Structured Output with Function Calling
# =============================================================================
def structured_output_demo():
    """
    Extract structured data from unstructured text using function calling.
    This is useful for parsing information into a specific format.
    """
    print("\n=== Structured Output Demo ===\n")
    
    from langchain_core.pydantic_v1 import BaseModel, Field
    
    # Define the structure we want
    class PersonInfo(BaseModel):
        """Information about a person."""
        name: str = Field(description="The person's full name")
        age: int = Field(description="The person's age in years")
        occupation: str = Field(description="The person's job or occupation")
        hobbies: list[str] = Field(description="List of the person's hobbies")
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    # Create structured output chain
    structured_llm = llm.with_structured_output(PersonInfo)
    
    # Test with unstructured text
    text = """
    John Smith is a 32-year-old software engineer who works at a tech startup.
    In his free time, he enjoys hiking, playing guitar, and reading sci-fi novels.
    """
    
    result = structured_llm.invoke(text)
    
    print("Input text:", text)
    print("\nExtracted structured data:")
    print(f"Name: {result.name}")
    print(f"Age: {result.age}")
    print(f"Occupation: {result.occupation}")
    print(f"Hobbies: {', '.join(result.hobbies)}\n")
    
    return structured_llm

# =============================================================================
# 4. Multi-Chain Router
# =============================================================================
def create_router_chain():
    """
    Route queries to different specialized chains based on the question type.
    This is useful for handling different types of requests.
    """
    print("\n=== Router Chain Demo ===\n")
    
    from langchain_core.prompts import ChatPromptTemplate
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    # Create specialized prompts
    math_prompt = ChatPromptTemplate.from_template(
        "You are a math expert. Solve this problem step by step:\n{question}"
    )
    
    coding_prompt = ChatPromptTemplate.from_template(
        "You are a coding expert. Provide a clear code solution:\n{question}"
    )
    
    general_prompt = ChatPromptTemplate.from_template(
        "Answer this question helpfully:\n{question}"
    )
    
    # Create chains
    math_chain = math_prompt | llm | StrOutputParser()
    coding_chain = coding_prompt | llm | StrOutputParser()
    general_chain = general_prompt | llm | StrOutputParser()
    
    # Simple router function
    def route_question(question: str):
        question_lower = question.lower()
        if any(word in question_lower for word in ["calculate", "math", "sum", "multiply"]):
            return "math"
        elif any(word in question_lower for word in ["code", "program", "function", "python"]):
            return "coding"
        else:
            return "general"
    
    # Route and execute
    questions = [
        "Calculate the sum of squares from 1 to 10",
        "Write a Python function to reverse a string",
        "What is the capital of France?"
    ]
    
    for question in questions:
        route = route_question(question)
        print(f"Question: {question}")
        print(f"Routed to: {route}")
        
        if route == "math":
            answer = math_chain.invoke({"question": question})
        elif route == "coding":
            answer = coding_chain.invoke({"question": question})
        else:
            answer = general_chain.invoke({"question": question})
        
        print(f"Answer: {answer}\n")

# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("ADVANCED LANGCHAIN IMPLEMENTATION")
    print("=" * 70)
    
    # Run all demonstrations
    rag_chain = create_rag_system()
    conv_chain, memory = create_conversational_chain()
    structured_llm = structured_output_demo()
    create_router_chain()
    
    print("\n" + "=" * 70)
    print("All demonstrations completed!")
    print("=" * 70)