from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import json

# Load environment variables
load_dotenv()

# =============================================================================
# 1. Advanced RAG with Re-ranking and Multi-Query
# =============================================================================
def advanced_rag_system():
    """
    Advanced RAG that:
    1. Generates multiple query variations
    2. Retrieves documents for each variation
    3. Re-ranks results by relevance
    4. Synthesizes a final answer
    """
    print("\n" + "="*70)
    print("1. ADVANCED RAG WITH MULTI-QUERY AND RE-RANKING")
    print("="*70 + "\n")
    
    # Sample knowledge base
    documents = [
        Document(page_content="Machine learning is a subset of AI that enables systems to learn from data."),
        Document(page_content="Deep learning uses neural networks with multiple layers to learn complex patterns."),
        Document(page_content="Supervised learning requires labeled data for training models."),
        Document(page_content="Unsupervised learning finds patterns in data without labels."),
        Document(page_content="Reinforcement learning trains agents through rewards and penalties."),
        Document(page_content="Transfer learning reuses pre-trained models for new tasks."),
        Document(page_content="Natural Language Processing (NLP) enables computers to understand human language."),
        Document(page_content="Computer vision allows machines to interpret visual information from images."),
    ]
    
    # Create vector store
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    splits = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    # Step 1: Generate multiple query variations
    query_generation_prompt = ChatPromptTemplate.from_template(
        """You are an AI assistant that generates search queries. 
        Given the original question, generate 3 different variations that capture different aspects.
        
        Original question: {question}
        
        Return ONLY a JSON array of strings, like: ["query1", "query2", "query3"]
        """
    )
    
    query_chain = query_generation_prompt | llm | StrOutputParser()
    
    def generate_queries(question: str) -> List[str]:
        """Generate multiple query variations"""
        result = query_chain.invoke({"question": question})
        try:
            queries = json.loads(result)
            return [question] + queries  # Include original
        except:
            return [question]
    
    # Step 2: Retrieve documents for all queries
    def retrieve_for_queries(queries: List[str]) -> List[Document]:
        """Retrieve documents for multiple queries and deduplicate"""
        all_docs = []
        seen_content = set()
        
        for query in queries:
            docs = retriever.invoke(query)
            for doc in docs:
                if doc.page_content not in seen_content:
                    all_docs.append(doc)
                    seen_content.add(doc.page_content)
        
        return all_docs
    
    # Step 3: Re-rank documents by relevance
    rerank_prompt = ChatPromptTemplate.from_template(
        """Given the question and a document, rate the relevance from 0-10.
        
        Question: {question}
        Document: {document}
        
        Return ONLY a number between 0 and 10.
        """
    )
    
    rerank_chain = rerank_prompt | llm | StrOutputParser()
    
    def rerank_documents(question: str, docs: List[Document]) -> List[Document]:
        """Re-rank documents by relevance score"""
        scored_docs = []
        
        for doc in docs:
            try:
                score = float(rerank_chain.invoke({
                    "question": question,
                    "document": doc.page_content
                }))
                scored_docs.append((score, doc))
            except:
                scored_docs.append((5.0, doc))  # Default score
        
        # Sort by score descending and return top 3
        scored_docs.sort(reverse=True, key=lambda x: x[0])
        return [doc for score, doc in scored_docs[:3]]
    
    # Step 4: Generate final answer
    answer_prompt = ChatPromptTemplate.from_template(
        """Answer the question based on the following context. Be comprehensive and detailed.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:
        """
    )
    
    # Combine everything into a chain
    def rag_pipeline(question: str) -> str:
        print(f"Original Question: {question}\n")
        
        # Generate query variations
        queries = generate_queries(question)
        print(f"Generated {len(queries)} query variations:")
        for i, q in enumerate(queries, 1):
            print(f"  {i}. {q}")
        print()
        
        # Retrieve documents
        docs = retrieve_for_queries(queries)
        print(f"Retrieved {len(docs)} unique documents\n")
        
        # Re-rank
        ranked_docs = rerank_documents(question, docs)
        print(f"Top 3 documents after re-ranking:")
        for i, doc in enumerate(ranked_docs, 1):
            print(f"  {i}. {doc.page_content[:60]}...")
        print()
        
        # Generate answer
        context = "\n\n".join([doc.page_content for doc in ranked_docs])
        answer_chain = answer_prompt | llm | StrOutputParser()
        answer = answer_chain.invoke({"context": context, "question": question})
        
        return answer
    
    # Test it
    question = "How do machines learn from data?"
    answer = rag_pipeline(question)
    print(f"Final Answer:\n{answer}\n")

# =============================================================================
# 2. Self-Correcting Chain with Validation
# =============================================================================
def self_correcting_chain():
    """
    A chain that validates its own output and self-corrects if needed.
    This demonstrates iterative improvement and quality control.
    """
    print("\n" + "="*70)
    print("2. SELF-CORRECTING CHAIN WITH VALIDATION")
    print("="*70 + "\n")
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    
    # Define expected output structure
    class MathProblemSolution(BaseModel):
        """Structure for math problem solutions"""
        problem: str = Field(description="The original problem")
        steps: List[str] = Field(description="Step-by-step solution")
        answer: str = Field(description="Final numerical answer")
        verification: str = Field(description="Verification of the answer")
    
    # Step 1: Initial solution generation
    solution_prompt = ChatPromptTemplate.from_template(
        """Solve this math problem step by step:
        
        Problem: {problem}
        
        Provide a detailed solution with clear steps.
        """
    )
    
    solution_chain = solution_prompt | llm | StrOutputParser()
    
    # Step 2: Validation
    validation_prompt = ChatPromptTemplate.from_template(
        """Review this math solution for correctness:
        
        Problem: {problem}
        Solution: {solution}
        
        Is this solution correct? Respond with:
        - "CORRECT" if the solution is right
        - "INCORRECT: [explanation]" if there are errors
        """
    )
    
    validation_chain = validation_prompt | llm | StrOutputParser()
    
    # Step 3: Correction (if needed)
    correction_prompt = ChatPromptTemplate.from_template(
        """The previous solution was incorrect. Here's why:
        {validation_result}
        
        Problem: {problem}
        Previous attempt: {solution}
        
        Provide a CORRECTED solution:
        """
    )
    
    correction_chain = correction_prompt | llm | StrOutputParser()
    
    # Self-correcting pipeline
    def solve_with_validation(problem: str, max_attempts: int = 3) -> str:
        print(f"Problem: {problem}\n")
        
        solution = None
        for attempt in range(1, max_attempts + 1):
            print(f"--- Attempt {attempt} ---")
            
            if attempt == 1:
                solution = solution_chain.invoke({"problem": problem})
            else:
                solution = correction_chain.invoke({
                    "problem": problem,
                    "solution": solution,
                    "validation_result": validation_result
                })
            
            print(f"Solution:\n{solution}\n")
            
            # Validate
            validation_result = validation_chain.invoke({
                "problem": problem,
                "solution": solution
            })
            
            print(f"Validation: {validation_result}\n")
            
            if validation_result.strip().upper().startswith("CORRECT"):
                print("✓ Solution validated successfully!\n")
                return solution
        
        print("⚠ Max attempts reached. Returning last solution.\n")
        return solution
    
    # Test it
    problem = "If a train travels 120 km in 2 hours, then stops for 30 minutes, then travels another 90 km in 1.5 hours, what is its average speed for the entire journey?"
    final_solution = solve_with_validation(problem)

# =============================================================================
# 3. Parallel Chain Execution with Aggregation
# =============================================================================
def parallel_chain_execution():
    """
    Execute multiple chains in parallel and aggregate results.
    Useful for getting diverse perspectives or analysis.
    """
    print("\n" + "="*70)
    print("3. PARALLEL CHAIN EXECUTION WITH AGGREGATION")
    print("="*70 + "\n")
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    
    # Create different analysis chains
    technical_prompt = ChatPromptTemplate.from_template(
        """Analyze this from a TECHNICAL perspective:
        
        Topic: {topic}
        
        Focus on technical details, implementation, and specifications.
        """
    )
    
    business_prompt = ChatPromptTemplate.from_template(
        """Analyze this from a BUSINESS perspective:
        
        Topic: {topic}
        
        Focus on market impact, ROI, and business value.
        """
    )
    
    user_prompt = ChatPromptTemplate.from_template(
        """Analyze this from a USER perspective:
        
        Topic: {topic}
        
        Focus on user experience, usability, and accessibility.
        """
    )
    
    # Create chains
    technical_chain = technical_prompt | llm | StrOutputParser()
    business_chain = business_prompt | llm | StrOutputParser()
    user_chain = user_prompt | llm | StrOutputParser()
    
    # Parallel execution
    parallel_chain = RunnableParallel(
        technical=technical_chain,
        business=business_chain,
        user=user_chain
    )
    
    # Aggregation chain
    aggregation_prompt = ChatPromptTemplate.from_template(
        """Synthesize these three perspectives into a comprehensive analysis:
        
        Technical Perspective:
        {technical}
        
        Business Perspective:
        {business}
        
        User Perspective:
        {user}
        
        Provide a balanced, integrated analysis that considers all viewpoints:
        """
    )
    
    aggregation_chain = aggregation_prompt | llm | StrOutputParser()
    
    # Full pipeline
    full_chain = parallel_chain | aggregation_chain
    
    # Test it
    topic = "Implementing AI-powered chatbots in customer service"
    
    print(f"Topic: {topic}\n")
    print("Executing parallel analysis...\n")
    
    result = full_chain.invoke({"topic": topic})
    
    print("Integrated Analysis:")
    print(result)
    print()

# =============================================================================
# 4. Dynamic Prompt Engineering with Few-Shot Learning
# =============================================================================
def dynamic_few_shot_learning():
    """
    Dynamically select relevant examples for few-shot learning based on the query.
    This improves accuracy by showing the model similar examples.
    """
    print("\n" + "="*70)
    print("4. DYNAMIC FEW-SHOT LEARNING")
    print("="*70 + "\n")
    
    # Example database (in production, this would be much larger)
    examples = [
        {
            "input": "I can't log into my account",
            "category": "authentication",
            "sentiment": "frustrated",
            "response": "I understand login issues are frustrating. Let's verify your credentials and reset if needed."
        },
        {
            "input": "Your product is amazing!",
            "category": "feedback",
            "sentiment": "positive",
            "response": "Thank you so much! We're thrilled you're enjoying our product."
        },
        {
            "input": "How do I cancel my subscription?",
            "category": "billing",
            "sentiment": "neutral",
            "response": "I can help you with that. You can cancel anytime from your account settings."
        },
        {
            "input": "This feature doesn't work as expected",
            "category": "bug_report",
            "sentiment": "concerned",
            "response": "Thank you for reporting this. Let me document the issue and investigate."
        },
        {
            "input": "When will feature X be available?",
            "category": "feature_request",
            "sentiment": "curious",
            "response": "Great question! Feature X is on our roadmap for Q2. I'll add your interest."
        }
    ]
    
    # Create embeddings for examples
    embeddings = OpenAIEmbeddings()
    example_texts = [ex["input"] for ex in examples]
    example_embeddings = embeddings.embed_documents(example_texts)
    
    # Store in FAISS
    example_docs = [Document(page_content=ex["input"], metadata=ex) for ex in examples]
    example_vectorstore = FAISS.from_documents(example_docs, embeddings)
    
    # Retriever for similar examples
    example_retriever = example_vectorstore.as_retriever(search_kwargs={"k": 2})
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    
    # Dynamic few-shot prompt
    def create_dynamic_prompt(user_input: str) -> str:
        """Create a prompt with relevant examples"""
        similar_examples = example_retriever.invoke(user_input)
        
        prompt = """You are a customer service AI. Respond professionally and empathetically.

Here are examples of good responses:

"""
        
        for i, doc in enumerate(similar_examples, 1):
            metadata = doc.metadata
            prompt += f"""Example {i}:
Customer: {metadata['input']}
Category: {metadata['category']}
Sentiment: {metadata['sentiment']}
Agent: {metadata['response']}

"""
        
        prompt += f"""Now respond to this customer:
Customer: {user_input}

Agent:"""
        
        return prompt
    
    # Test queries
    test_queries = [
        "I'm having trouble accessing my dashboard",
        "Your service is terrible and slow!",
        "Do you have an API I can integrate with?"
    ]
    
    for query in test_queries:
        print(f"Customer: {query}\n")
        
        # Get similar examples
        similar = example_retriever.invoke(query)
        print("Similar examples retrieved:")
        for i, doc in enumerate(similar, 1):
            print(f"  {i}. {doc.metadata['input']} (Category: {doc.metadata['category']})")
        print()
        
        # Generate response
        dynamic_prompt = create_dynamic_prompt(query)
        response = llm.invoke(dynamic_prompt)
        
        print(f"Agent: {response.content}\n")
        print("-" * 70 + "\n")

# =============================================================================
# 5. Conversational Agent with Tools and Planning
# =============================================================================
def conversational_agent_with_tools():
    """
    An agent that can use tools, plan multi-step actions, and maintain conversation.
    This simulates a more intelligent assistant.
    """
    print("\n" + "="*70)
    print("5. CONVERSATIONAL AGENT WITH TOOLS AND PLANNING")
    print("="*70 + "\n")
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    # Define available tools (simulated)
    tools_description = """
Available Tools:
1. calculate(expression: str) -> float
   Evaluates mathematical expressions
   Example: calculate("25 * 4 + 10")

2. search_knowledge(query: str) -> str
   Searches a knowledge base
   Example: search_knowledge("Python list methods")

3. get_current_time() -> str
   Returns current date and time
   Example: get_current_time()
"""
    
    # Simulate tool execution
    def execute_tool(tool_name: str, args: str) -> str:
        """Simulate tool execution"""
        if tool_name == "calculate":
            try:
                result = eval(args.strip('"\''))
                return f"Result: {result}"
            except:
                return "Error: Invalid expression"
        elif tool_name == "search_knowledge":
            return f"Knowledge result for '{args}': [Simulated search results]"
        elif tool_name == "get_current_time":
            from datetime import datetime
            return f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        else:
            return f"Error: Unknown tool '{tool_name}'"
    
    # Agent prompt with planning
    agent_prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a helpful AI assistant with access to tools.

{tools_description}

When you need to use a tool, respond with:
TOOL: tool_name(arguments)

When you have the final answer, respond with:
ANSWER: your response

Think step-by-step and plan your actions."""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    chain = agent_prompt | llm | StrOutputParser()
    
    # Create conversation memory
    store = {}
    
    def get_session_history(session_id: str):
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]
    
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )
    
    # Agent loop
    def run_agent(user_input: str, session_id: str = "default", max_steps: int = 5):
        print(f"User: {user_input}\n")
        
        for step in range(max_steps):
            response = chain_with_history.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            
            print(f"Agent (Step {step + 1}): {response}\n")
            
            # Check if using a tool
            if response.strip().startswith("TOOL:"):
                tool_call = response.strip()[5:].strip()
                
                # Parse tool call
                if "(" in tool_call:
                    tool_name = tool_call[:tool_call.index("(")]
                    args = tool_call[tool_call.index("(")+1:tool_call.rindex(")")]
                    
                    # Execute tool
                    result = execute_tool(tool_name, args)
                    print(f"Tool Result: {result}\n")
                    
                    # Continue with tool result
                    user_input = f"Tool result: {result}. Continue with your task."
                else:
                    print("Error: Invalid tool call format\n")
                    break
            
            # Check if final answer
            elif response.strip().startswith("ANSWER:"):
                final_answer = response.strip()[7:].strip()
                print(f"Final Answer: {final_answer}\n")
                break
        
        return response
    
    # Test scenarios
    test_queries = [
        "What is 15% of 240?",
        "Calculate (50 + 30) * 2 and then add 100 to the result"
    ]
    
    for i, query in enumerate(test_queries):
        print(f"\n{'='*70}")
        print(f"Scenario {i+1}")
        print(f"{'='*70}\n")
        run_agent(query, session_id=f"test_{i}")

# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("SUPER ADVANCED LANGCHAIN IMPLEMENTATION")
    print("="*70)
    
    # Run all demonstrations
    advanced_rag_system()
    self_correcting_chain()
    parallel_chain_execution()
    dynamic_few_shot_learning()
    conversational_agent_with_tools()
    
    print("\n" + "="*70)
    print("ALL DEMONSTRATIONS COMPLETED!")
    print("="*70 + "\n")