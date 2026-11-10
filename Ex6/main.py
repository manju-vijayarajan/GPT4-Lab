import requests
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:0.5b"
knowledge_base = {
    "cold": {
        "keywords": ["sneeze", "runny nose", "congestion", "cough"],
        "advice": "This may be a common cold. Rest, drink warm fluids, and monitor your symptoms."
    },
    "flu": {
        "keywords": ["fever", "body ache", "chills", "fatigue"],
        "advice": "These symptoms may be associated with flu. Stay hydrated, get adequate rest, and monitor your temperature."
    },
    "migraine": {
        "keywords": ["headache", "nausea", "sensitivity to light", "sensitivity to sound"],
        "advice": "These symptoms may be associated with a migraine. Rest in a quiet, dark room and monitor your symptoms."
    },
    "indigestion": {
        "keywords": ["stomach pain", "nausea", "bloating", "heartburn"],
        "advice": "These symptoms may be related to indigestion. Avoid spicy foods, drink water, and eat light meals."
    }
}
def check_knowledge_base(symptoms):
    symptoms = symptoms.lower()
    for condition, data in knowledge_base.items():
        for keyword in data["keywords"]:
            if keyword in symptoms:
                return condition, data["advice"]

    return None, None
def ollama_response(symptoms):
    prompt = f"""
You are an educational medical assistant.

User symptoms:
{symptoms}

Provide:
1. Possible common causes or conditions
2. Basic home-care measures
3. When the person should consult a healthcare professional

Do not claim to provide a diagnosis.
Always include a medical disclaimer.
Keep the response concise and easy to understand.
"""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        return f"Unable to connect to Ollama: {e}"
def medical_assistant(symptoms):
    condition, advice = check_knowledge_base(symptoms)
    if condition:
        return (
            f"Possible condition: {condition.title()}\n\n"
            f"Basic advice: {advice}\n\n"
            "Disclaimer: This is only general educational information "
            "and is not a medical diagnosis. Consult a qualified "
            "healthcare professional for proper evaluation."
        )
    return ollama_response(symptoms)
print("      AI-Powered Medical Assistant")
while True:
    symptoms = input("Enter your symptoms: ").strip()
    if symptoms.lower() == "exit":
        print("\nThank you. Stay healthy!")
        break
    if not symptoms:
        print("Please enter your symptoms.\n")
        continue
    print("\nAssistant:")
    print(medical_assistant(symptoms))
    print("\n" + "-" * 50 + "\n")