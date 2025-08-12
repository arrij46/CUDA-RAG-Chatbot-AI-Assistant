import requests
import speech_recognition as sr

API_KEY = "Your key"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama3-70b-8192"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def chat_with_groq(messages):
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.7
    }

    response = requests.post(GROQ_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        reply = response.json()["choices"][0]["message"]["content"]
        return reply[:200]  # Truncate to 200 characters
    else:
        print("Error:", response.status_code, response.text)
        return "Sorry, something went wrong."

def recognize_speech():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    print("🎤 Speak now...")

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        print("Sorry, could not understand audio.")
        return None
    except sr.RequestError as e:
        print(f"Speech recognition service error: {e}")
        return None

def main():
    print("🤖 Groq LLaMA3 Terminal Chatbot with Voice Input")
    print("Type 'exit' or say 'exit' to quit.\n")

    # System prompt with character constraint
    messages = [{
        "role": "system",
        "content": "You are a helpful assistant. Keep all responses strictly under 200 characters."
    }]

    while True:
        print("🎙️ Press Enter to speak your query (or type it instead):")
        typed_input = input("> ").strip()

        if typed_input.lower() == "exit":
            print("Goodbye!")
            break
        elif typed_input:
            user_input = typed_input
        else:
            user_input = recognize_speech()
            if user_input is None or user_input.lower() == "exit":
                print("Goodbye!")
                break

        # Reinforce the length constraint in every user prompt
        messages.append({
            "role": "user",
            "content": f"{user_input}\n(Respond in under 200 characters only.)"
        })

        reply = chat_with_groq(messages)
        messages.append({"role": "assistant", "content": reply})
        print(f"Bot: {reply}\n")

if __name__ == "__main__":
    main()
