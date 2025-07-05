import os
import time
import re
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI
from groq import Groq
import chromadb
from collections import defaultdict, deque
import logging
import azure.cognitiveservices.speech as speechsdk
import replicate

# ------------------- Setup & Globals -------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [APP] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

load_dotenv()
openai_key = os.getenv("OPEN_AI_KEY")
groq_key = os.getenv("GROQ_API_KEY")
azure_speech_key = os.getenv("AZURE_SPEECH_KEY")
azure_speech_region = os.getenv("AZURE_SPEECH_REGION")
replicate_api_token = os.getenv("REPLICATE_API_TOKEN")

client = OpenAI(api_key=openai_key)
client2 = Groq(api_key=groq_key)
os.environ["REPLICATE_API_TOKEN"] = replicate_api_token

chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection = chroma_client.get_collection("document_qa_collection")

CONTEXT_WINDOW = 6
user_histories = defaultdict(lambda: deque(maxlen=CONTEXT_WINDOW * 2))

app = Flask(__name__)
AUDIO_DIR = "audio"
STATIC_AUDIO_FILENAME = "chat_response.mp3"
STATIC_AUDIO_PATH = os.path.join(AUDIO_DIR, STATIC_AUDIO_FILENAME)
os.makedirs(AUDIO_DIR, exist_ok=True)

# ------------------- Helper Functions -------------------

def query_documents(question, n_results=5):
    logger.info("Querying ChromaDB for relevant context...")
    results = collection.query(query_texts=[question], n_results=n_results)
    return [doc for sublist in results["documents"] for doc in sublist]

def synthesize_speech(text, voice="en-US-AriaNeural"):
    logger.info("Generating expressive TTS audio using Azure Speech with SSML...")
    try:
        if not text.strip():
            logger.warning("TTS skipped: empty input text.")
            return None, 0

        # Compose SSML markup for expressive speech
        ssml_string = f"""
        <speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'>
            <voice name='{voice}'>
                {text}
            </voice>
        </speak>
        """

        speech_config = speechsdk.SpeechConfig(subscription=azure_speech_key, region=azure_speech_region)
        speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
        )
        audio_config = speechsdk.audio.AudioOutputConfig(filename=STATIC_AUDIO_PATH)
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
        
        # Try expressive SSML first
        result = synthesizer.speak_ssml_async(ssml_string).get()
        
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            logger.info(f"✅ Expressive audio saved to: {STATIC_AUDIO_PATH}")
            plain_text = re.sub('<[^>]*>', '', text)
            approx_duration = len(plain_text.split()) / 2.5
            return STATIC_AUDIO_FILENAME, round(approx_duration, 2)
        else:
            logger.error(f"SSML synthesis failed: {result.reason}. Falling back to plain text.")
            plain_text = re.sub('<[^>]*>', '', text)
            result = synthesizer.speak_text_async(plain_text).get()
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                logger.info(f"✅ Fallback plain text audio saved to: {STATIC_AUDIO_PATH}")
                approx_duration = len(plain_text.split()) / 2.5
                return STATIC_AUDIO_FILENAME, round(approx_duration, 2)
            else:
                logger.error(f"Fallback TTS also failed: {result.reason}")
                return None, 0

    except Exception as e:
        logger.error(f"TTS generation failed entirely: {e}")
        return None, 0

def build_messages(user_name, selected_module, history, question, n_context=5):
    relevant_chunks = query_documents(question)
    context = "\n\n".join(relevant_chunks)
    system_prompt = f"""You are Eli the Elephant, a friendly and wise study buddy for children (ages 5-10).
Your goal is to make learning a fun, continuous conversation with {user_name} about {selected_module}.
1.  **Use Simple Language:** Talk like you're speaking to a 7-year-old.
2.  **Be a Conversational Buddy:** This is an ongoing chat.
3.  **Write for Voice and for Reading:** Use enthusiastic words and punctuation (like "..." for pauses) to create a warm, expressive tone.
4.  **Stay in Character:** Always be a positive, safe, and encouraging elephant buddy, never an AI. Stick strictly to the `{{context}}`.
5.  **Be Brief:** Your entire response must be **less than 750 characters**.
6.  **Suggest the Next Step:** ALWAYS end your response by asking a related, exciting follow-up question to guide the conversation. Make it an invitation to learn more.
7.  **Use Expressive Speech:** To make your voice sound more exciting, wrap key words in `<emphasis>` tags for emphasis. For a dramatic pause, use `<break time="400ms"/>`.

**Example of a perfect response:**
"Wow, that's a great question! Bees buzz to flap their wings super, <emphasis>super</emphasis> fast! <break time="300ms"/> Isn't that neat? I wonder... should we learn about how they make honey next?"

---
**Context:**
{context}
"""
    messages = [{"role": "system", "content": system_prompt}]
    history_list = list(history)
    for role, content in history_list[-n_context*2:]:
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})
    return messages

def generate_image_prompt(question, answer):
    logger.info("Generating a dedicated image prompt...")
    try:
        # Remove SSML tags for clarity
        plain_answer = re.sub('<[^>]*>', '', answer)
        prompt_generation_messages = [
            {
                "role": "system",
                "content": (
                "You are an expert at creating prompts for a text-to-image AI. "
                "Your task is to summarize a user's question and an assistant's answer "
                "into a short, descriptive prompt for an image. "
                "The image style MUST be 'pixel art, cute and friendly children's book illustration'. "
                "The prompt should describe a scene that visually represents the main subject or concept discussed, based on the question and answer. "
                "Do NOT include any text, letters, or writing in the image. "
                "Do NOT mention any specific character or animal unless it is directly relevant to the question or answer. "
                "Focus on making the image context-based and visually engaging for children. "
                "**Output ONLY the prompt text, with no preamble, quotes, or explanation.** "
                "Example Input: "
                "Question: 'Why do volcanoes erupt?' "
                "Answer: 'Volcanoes erupt when hot melted rock called magma pushes up and bursts out of the earth!' "
                "Example Output: "
                "pixel art, a volcano erupting with bright lava and smoke, children watching from a safe distance, cute and friendly children's book illustration style."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Summarize this for an image prompt:\n\n"
                    f"Question: \"{question}\"\n\n"
                    f"Answer: \"{plain_answer}\""
                )
            }
        ]
        response = client2.chat.completions.create(
            model="gemma-7b-it",  # <<< FASTER, SMALLER MODEL
            messages=prompt_generation_messages,
            max_tokens=40,         # <<< Use less tokens (prompt will be short)
            temperature=0.5,
            top_p=1,
        )
        image_prompt = response.choices[0].message.content.strip().replace("\"", "")
        logger.info(f"Generated image prompt: '{image_prompt}'")
        return image_prompt
    except Exception as e:
        logger.error(f"Failed to generate image prompt: {e}")
        # Always fallback with pixel art + main subject
        return f"pixel art, a scene representing the topic of '{question}', cute and friendly children's book illustration"


def generate_image(prompt, aspect_ratio="16:9"):
    try:
        input = { "prompt": prompt.strip(), "aspect_ratio": aspect_ratio, "go_fast": True, "num_outputs": 1, "output_format": "png", "output_quality": 80 }
        logger.info(f"Calling Replicate for image: prompt=\"{input['prompt']}\"")
        output = replicate.run("black-forest-labs/flux-schnell", input=input)
        if hasattr(output, "__iter__") and not isinstance(output, str):
            for item in output:
                if isinstance(item, str) and item.startswith("http"):
                    logger.info(f"Image generated: {item}")
                    return item
                if hasattr(item, "url") and isinstance(item.url, str):
                    logger.info(f"Image generated: {item.url}")
                    return item.url
        elif isinstance(output, str) and output.startswith("http"):
            logger.info(f"Image generated: {output}")
            return output
        logger.error(f"Replicate image output did not contain a valid URL. Raw output: {output}")
        return None
    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        return None

# ------------------- Flask Endpoints -------------------

@app.route("/audio/<filename>")
def serve_audio(filename):
    return send_from_directory(AUDIO_DIR, filename, mimetype="audio/mpeg")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    question = data.get("question", "").strip()
    user_name = data.get("user_name", "friend").strip()
    selected_module = data.get("selected_module", "something").strip()
    if not question:
        return jsonify({"error": "Empty question"}), 400

    try:
        history = user_histories[user_name]
        history.append(('user', question))
        messages = build_messages(user_name, selected_module, history, question, n_context=CONTEXT_WINDOW)
        start_time = time.time()
        
        response_stream = client2.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=150,
            temperature=0.2,
            top_p=0.9,
            stream=True
        )
        full_answer = "".join(chunk.choices[0].delta.content for chunk in response_stream if chunk.choices[0].delta and chunk.choices[0].delta.content)
        history.append(('assistant', full_answer))
        logger.info(f"Generated response: {full_answer.strip()[:100]}...")

        audio_filename, audio_duration = synthesize_speech(full_answer)
        image_gen_prompt = generate_image_prompt(question, full_answer)
        image_url = generate_image(image_gen_prompt, aspect_ratio="16:9")
        elapsed = round(time.time() - start_time, 2)

        response_data = {
            "answer": full_answer,
            "audio_url": f"http://localhost:5000/audio/{audio_filename}" if audio_filename else None,
            "duration": audio_duration,
            "image_url": image_url if image_url else None
        }

        logger.info(f"✅ Response ready in {elapsed}s")
        return jsonify(response_data), 200

    except Exception as e:
        logger.exception("Chat processing error")
        return jsonify({"error": str(e)}), 500

# ------------------- Main -------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)



