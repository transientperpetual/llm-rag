from langchain_ollama import OllamaLLM

# Initialize the LLaMA model
llm = OllamaLLM(model="llama3.2")

# Test with a sample prompt
response = llm.invoke("Who owned the Banner in The Fountainhead?")
print(response)