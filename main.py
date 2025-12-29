from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env file
load_dotenv()

# Initialize the model (automatically uses OPENAI_API_KEY from .env)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# Create a prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that answers questions concisely."),
    ("user", "{question}")
])

# Create output parser
output_parser = StrOutputParser()

# Chain them together using LCEL (LangChain Expression Language)
chain = prompt | llm | output_parser

# Use the chain
if __name__ == "__main__":
    # Example 1: Simple question
    response = chain.invoke({"question": "What is the biggest city in the world?"})
    print(f"Response: {response}\n")
    
    # Example 2: Stream responses
    print("Streaming response:")
    for chunk in chain.stream({"question": "Tell me a short story"}):
        print(chunk, end="", flush=True)
    print("\n")
    
    # Example 3: Batch processing
    questions = [
        {"question": "What is 4**7?"},
        {"question": "is Sauron a maiar"}
    ]
    responses = chain.batch(questions)
    print("\nBatch responses:")
    for q, r in zip(questions, responses):
        print(f"Q: {q['question']}\nA: {r}\n")