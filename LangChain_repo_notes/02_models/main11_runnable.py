from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

load_dotenv()

# Initialize model
llm = init_chat_model("google_genai:gemini-2.5-flash-lite", temperature=0.3)

print("=== Basic Runnable Chain ===")

# 1. Create prompt template (Runnable)
prompt = ChatPromptTemplate.from_template("Explain {topic} in simple terms.")

# 2. Chain prompt + model (using | operator)
basic_chain = prompt | llm

# 3. Run it
response = basic_chain.invoke({"topic": "blockchain"}) # Explain {blockchain} in simple terms.
print(f"Response: {response.content[:100]}...")
print()

print("=== Chain with Output Parser ===")

# Add string output parser to the chain
parser = StrOutputParser()
chain_with_parser = prompt | llm | parser

# This returns just the string content, not the full message object
result = chain_with_parser.invoke({"topic": "AI"})
print(f"Parsed result: {result[:100]}...")
print()

print("=== Custom Function in Chain ===")

# Custom function as Runnable
def make_uppercase(text):
    return text.upper()

uppercase_runnable = RunnableLambda(make_uppercase)

# Chain that includes custom function
chain_with_custom = prompt | llm | parser | uppercase_runnable

result = chain_with_custom.invoke({"topic": "python"})
print(f"Uppercase result: {result[:100]}...")
print()

print("=== Parallel Processing ===")

# Create different prompts for parallel execution
joke_prompt = ChatPromptTemplate.from_template("Tell a short joke about {topic}")
fact_prompt = ChatPromptTemplate.from_template("Give one interesting fact about {topic}")

# Run them in parallel using RunnableParallel
from langchain_core.runnables import RunnableParallel

parallel_chain = RunnableParallel(
    joke=joke_prompt | llm | parser,
    fact=fact_prompt | llm | parser
)

results = parallel_chain.invoke({"topic": "cats"})
print(f"Joke: {results['joke'][:80]}...")
print(f"Fact: {results['fact'][:80]}...")
print()

print("=== Streaming Example ===")

# Stream tokens as they come
print("Streaming response:")
for chunk in basic_chain.stream({"topic": "space"}):
    print(chunk.content, end="", flush=True)
print("\n")