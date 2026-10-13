
import os

from dotenv import load_dotenv
from openai import OpenAI
from google import genai
from huggingface_hub import InferenceClient

load_dotenv()

# Read API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")


# 1. GPT API
def ask_gpt(prompt):
    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.responses.create(
        model="gpt-4.1-mini",
        instructions=(
            "You are a friendly chatbot. Give short, natural "
            "answers. Reply briefly to greetings."
        ),
        input=prompt
    )

    return response.output_text


# 2. Gemini API
def ask_gemini(prompt):
    client = genai.Client(api_key=GEMINI_API_KEY)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=(
            "You are a friendly chatbot. Give short, natural "
            "answers. Reply briefly to greetings.\n\n"
            + prompt
        )
    )

    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")

    return response.text


# 3. Hugging Face API
def ask_huggingface(prompt):
    client = InferenceClient(api_key=HF_TOKEN)

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b:fastest",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a friendly chatbot. Give short, "
                    "natural answers. Reply briefly to greetings."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=150
    )

    answer = response.choices[0].message.content

    if not answer:
        raise RuntimeError("Hugging Face returned an empty response.")

    return answer


# Provider selection
def choose_model():
    print("\nChoose your AI model:")
    print("1. GPT")
    print("2. Gemini")
    print("3. Hugging Face")
    print("4. Exit")

    choice = input("Enter your choice: ").strip()

    models = {
        "1": ("GPT", ask_gpt, OPENAI_API_KEY),
        "2": ("Gemini", ask_gemini, GEMINI_API_KEY),
        "3": ("Hugging Face", ask_huggingface, HF_TOKEN)
    }

    if choice == "4":
        return None

    if choice not in models:
        print("Invalid choice. Please try again.")
        return "invalid"

    name, function, api_key = models[choice]

    if not api_key:
        print(f"{name} API key is missing from .env.")
        return "invalid"

    return name, function


# Main application
def main():
    print("=== MULTI-MODEL AI CHATBOT ===")

    selected = choose_model()

    while selected:
        if selected == "invalid":
            selected = choose_model()
            continue

        name, model_function = selected
        print(f"\nSelected model: {name}")
        print("Type /switch to change model.")
        print("Type /exit to close the chatbot.")

        while True:
            prompt = input("\nYou: ").strip()

            if prompt.lower() == "/exit":
                print("Goodbye!")
                return

            if prompt.lower() == "/switch":
                selected = choose_model()
                break

            if not prompt:
                print("Please enter a message.")
                continue

            try:
                answer = model_function(prompt)
                print(f"\n{name}: {answer}")

            except Exception as error:
                print(f"\n{name} API failed: {error}")
                print("Please choose another model.")

                selected = choose_model()
                break

        if selected is None:
            print("Goodbye!")
            return


if __name__ == "__main__":
    main()
