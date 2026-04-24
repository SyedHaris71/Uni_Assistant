from dotenv import load_dotenv
import os
import time

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load API Key

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print(" ERROR: GROQ_API_KEY not found. Check your .env file.")
    exit()

print(" API Key Loaded")

# LLM (FIXED MODEL)

llm = ChatGroq(
    model="llama-3.1-8b-instant",   # 
    temperature=0
)
# Memory (Manual)
memory = []
# Intent Classifier
intent_prompt = ChatPromptTemplate.from_template("""
Classify the user question into one of these categories:
fee, admission, hostel, scholarship, not_related

Question: {question}

Only return the category name.
""")

intent_chain = intent_prompt | llm | StrOutputParser()
# Main Prompt
main_prompt = ChatPromptTemplate.from_template("""
You are a helpful university assistant.

Rules:
- Answer only university-related questions
- Be short and clear
- Remember user details like name if provided

Conversation History:
{history}

User Question:
{question}

Answer:
""")

response_chain = main_prompt | llm | StrOutputParser()
# Error Handling (IMPROVED)
def safe_invoke(chain, inputs, retries=2):
    for attempt in range(retries):
        try:
            return chain.invoke(inputs)
        except Exception as e:
            print(f" Attempt {attempt+1} failed:", e)
            time.sleep(1)

    return "Sorry, the system is currently unavailable."
# CLI Chatbot
def main():
    print("=" * 50)
    print("FAST University AI Assistant")
    print("=" * 50)

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # 1. Intent classification
        intent = safe_invoke(intent_chain, {"question": user_input}).strip()

        # 2. Load memory
        history = "\n".join(memory)

        # 3. Generate response
        response = safe_invoke(
            response_chain,
            {
                "history": history,
                "question": user_input
            }
        )

        # 4. Save memory
        memory.append(f"You: {user_input}")
        memory.append(f"Bot: {response}")

        # 5. Output
        print(f"[{intent.upper()}] {response}")


if __name__ == "__main__":
    main()