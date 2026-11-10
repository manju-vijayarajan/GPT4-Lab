import requests
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:0.5b"
faq = {
    "pricing": "Our basic plan starts at $10/month with premium options.",
    "refund": "Refunds are available within 30 days of purchase. Please share your order ID.",
    "support": "You can reach support at support@example.com, 24/7."
}
def check_faq(email_text):
    for keyword, answer in faq.items():
        if keyword in email_text.lower():
            return answer
    return None
def ollama_response(prompt):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=120
        )
        if response.status_code == 200:
            return response.json().get("response", "").strip()
        return f"Error: {response.text}"
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to Ollama. Make sure Ollama is running."
    except requests.exceptions.Timeout:
        return "Error: Ollama request timed out."
def email_assistant(email_text):
    faq_answer = check_faq(email_text)
    if faq_answer:
        return "FAQ Reply: " + faq_answer
    prompt = f"""
You are an intelligent email assistant.

Draft a polite, professional, concise, and helpful reply
to the following incoming email.

Incoming Email:
{email_text}

Reply:
"""
    ai_answer = ollama_response(prompt)
    return "AI Reply: " + ai_answer

while True:
    email = input("Enter email: ")
    if email.lower().strip() == "exit":
        print("Exiting Email Assistant...")
        break
    if not email.strip():
        print("Please enter an email message.\n")
        continue
    reply = email_assistant(email)
    print("\nAssistant Reply:")
    print(reply)