import os
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables from .env files
load_dotenv()  # Load default .env file
load_dotenv('.env.local', override=True)  # Load .env.local and override any existing values

# Retrieve Azure OpenAI credentials from environment variables
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

if not api_key:
    raise ValueError("Missing AZURE_OPENAI_API_KEY in environment variables")
if not endpoint:
    raise ValueError("Missing AZURE_OPENAI_ENDPOINT in environment variables")
if not deployment:
    raise ValueError("Missing AZURE_OPENAI_DEPLOYMENT in environment variables")

def get_openai_client():
    return AzureOpenAI(
        api_key=api_key,
        api_version="2024-02-01",
        azure_endpoint=endpoint
    )

client = get_openai_client()

def construct_messages(system_text):
    messages=[
        {"role": "system", "content": system_text}
    ]
    return messages

def run_text_prompt_with_history(msgs, input_text, client=client):
    msgs.append({"role": "user", "content": [
            {"type": "text", "text": input_text},
        ]})
    response = client.chat.completions.create(
        model = deployment,
        messages = msgs,
        temperature=0.0,
    )
    msgs.append({"role": "assistant", "content": [
            {"type": "text", "text": response.choices[0].message.content},
        ]})
    return response.choices[0].message.content

def run_text_prompt(msgs, client=client):

    response = client.chat.completions.create(
        model = deployment,
        messages = msgs,
        temperature=0.0,
    )
    return response.choices[0].message.content


def get_gpt_response(message, client):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for booking travel trips."},
                {"role": "user", "content": message}
            ],
            temperature=0.7,  # Optional: Adjusts randomness of the output
        )

        # Print the assistant's reply
        reply = response.choices[0].message["content"]
        print("Assistant reply:", reply)
        return reply

    except Exception as e:
        print("Error calling OpenAI API:", e)
