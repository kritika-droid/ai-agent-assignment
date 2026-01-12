from agent import AIAgent

agent = AIAgent()

# Test 1: RAG question
response1 = agent.answer("What is the company leave policy?")
print("RAG Response:")
print(response1)
print("\n" + "-"*50 + "\n")

# Test 2: General question
response2 = agent.answer("What is Artificial Intelligence?")
print("Direct Response:")
print(response2)
