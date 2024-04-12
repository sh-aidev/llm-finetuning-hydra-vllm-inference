from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_response = client.chat.completions.create(
    model="sh-aidev/mistral-7b-v0.1-alpaca-chat",
    messages=[
        {"role": "system", "content": "List 3 historical events related to the following country"},
        {"role": "user", "content": "India"},
    ],
    max_tokens=200
)

print(chat_response.choices[0].message.content)
