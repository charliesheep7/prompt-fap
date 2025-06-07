import os
from flask import Flask, request, jsonify, render_template, session, send_file
import google.generativeai as genai
import requests
import json
from concurrent.futures import ThreadPoolExecutor
import threading
import uuid
from datetime import datetime
from dotenv import load_dotenv
import tempfile
import io

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-for-sessions')

# API Keys from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GROK_API_KEY = os.getenv('GROK_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Validate that all required API keys are present - but only fail at runtime, not import time
def validate_api_keys():
    if not OPENAI_API_KEY:
        raise ValueError("Missing required environment variable: OPENAI_API_KEY")

# Fapulous AI prompts for guided meditation after masturbation
FAPULOUS_PROMPTS = {
    "SYSTEM_PROMPT": """YYou are Fapulous-AI, a warm, evidence-based coach focused on men's post-orgasm recovery to help him acheive his goal: Feel less guilty after masturbation. Your tone is supportive, concise, and lightly humorous when appropriate. Cite the science in plain words once, **without academic jargon or hyperlinks**. Never moralise or shame; assume masturbation is normal. """,
    
    "QUESTION_1_PROMPT": """Acknowledge the user's mood in 1–2 lines. Normalize it using neuroscience. Avoid trying to "solve" anything yet — just validate and inform.

Keep it under 20 words.""",
    
    "QUESTION_2_PROMPT": """Respond with 1 short, confident line that tells the user you're collecting the best technique for that issue. Mention that the next session should make them feel x% better soon — make it light, not clinical.

Do not use breathwork, science, or affirmation yet.
Keep it under 20 words.""",
    
    "QUESTION_3_PROMPT": """Acknowledge the user's feeling with warmth and understanding. Tell them you have an affirmation card just for them, you want them to read it to themselves . but don't show the content. Be empathetic and encouraging.

Keep it under 20 words.""",
    
    "FINAL_CARD_PROMPT": """Based on user's goal from last response give a calm affirmation that starts with "I..." and stays under 20 words."""
}

# Configure clients
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/prompts', methods=['GET'])
def get_prompts():
    return jsonify({
        "fapulous_prompts": FAPULOUS_PROMPTS
    })

@app.route('/api/update-prompt', methods=['POST'])
def update_prompt():
    """Update a specific prompt"""
    data = request.get_json()
    prompt_key = data.get('prompt_key')
    new_prompt = data.get('new_prompt')
    
    if prompt_key in FAPULOUS_PROMPTS:
        FAPULOUS_PROMPTS[prompt_key] = new_prompt
        return jsonify({"success": True, "message": f"Updated {prompt_key}"})
    else:
        return jsonify({"success": False, "error": "Invalid prompt key"}), 400

@app.route('/api/fapulous-session', methods=['POST'])
def fapulous_session():
    """Handle the Fapulous guided meditation session"""
    # Validate API keys are present
    try:
        validate_api_keys()
    except ValueError as e:
        return jsonify({"error": str(e)}), 500
    
    data = request.get_json()
    step = data.get('step')  # 'start', 'question1', 'question2', 'reveal', 'final'
    user_message = data.get('user_message', '')
    session_data = data.get('session_data', {})
    
    if step == 'start':
        # Initial response when user clicks "I just fapped"
        prompt = FAPULOUS_PROMPTS["SYSTEM_PROMPT"] + "\n\n" + FAPULOUS_PROMPTS["QUESTION_1_PROMPT"]
        user_content = "The user just masturbated and wants guidance."
    elif step == 'question1':
        # Second response after user's first reply
        prompt = FAPULOUS_PROMPTS["SYSTEM_PROMPT"] + "\n\n" + FAPULOUS_PROMPTS["QUESTION_2_PROMPT"]
        user_content = f"User's first response: {user_message}"
    elif step == 'question2':
        # Third response - acknowledge feelings and mention affirmation card
        prompt = FAPULOUS_PROMPTS["SYSTEM_PROMPT"] + "\n\n" + FAPULOUS_PROMPTS["QUESTION_3_PROMPT"]
        user_content = f"User's second response: {user_message}"
        
        # Also generate the affirmation for later use
        affirmation_prompt = FAPULOUS_PROMPTS["SYSTEM_PROMPT"] + "\n\n" + FAPULOUS_PROMPTS["FINAL_CARD_PROMPT"]
        affirmation_content = f"User's goal/responses: {user_message}"
    else:
        return jsonify({"error": "Invalid step"}), 400
    
    def call_openai_chat(system_prompt, user_content):
        try:
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                "temperature": 0.7,
                "max_tokens": 100
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                assistant_message = data["choices"][0]["message"]["content"]
                return {
                    "success": True,
                    "response": assistant_message.strip(),
                    "error": None
                }
            else:
                return {
                    "success": False,
                    "response": None,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "response": None,
                "error": str(e)
            }
    
    # Get AI response
    result = call_openai_chat(prompt, user_content)
    
    if result["success"]:
        response_data = {
            "ai_response": result["response"],
            "step": step,
            "next_step": get_next_step(step),
            "session_data": session_data
        }
        
        # For question2, also generate the affirmation and store it
        if step == 'question2':
            affirmation_result = call_openai_chat(affirmation_prompt, affirmation_content)
            if affirmation_result["success"]:
                if 'stored_affirmation' not in session_data:
                    session_data['stored_affirmation'] = affirmation_result["response"]
                response_data["session_data"] = session_data
        
        # Add user message to session data for context
        if user_message:
            if 'conversation' not in session_data:
                session_data['conversation'] = []
            session_data['conversation'].append({
                "user": user_message,
                "ai": result["response"],
                "step": step
            })
            response_data["session_data"] = session_data
        
        return jsonify(response_data)
    else:
        return jsonify({"error": result["error"]}), 500

def get_next_step(current_step):
    """Get the next step in the conversation flow"""
    flow = {
        'start': 'question1',
        'question1': 'question2', 
        'question2': 'reveal',
        'final': 'complete'
    }
    return flow.get(current_step, 'complete')

@app.route('/api/transcribe-audio', methods=['POST'])
def transcribe_audio():
    """Transcribe audio file to text using OpenAI Whisper"""
    try:
        validate_api_keys()
    except ValueError as e:
        return jsonify({"error": str(e)}), 500
    
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio']
    
    if audio_file.filename == '':
        return jsonify({"error": "No audio file selected"}), 400
    
    try:
        # Create temporary file for audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_audio:
            audio_file.save(temp_audio.name)
            
            # Transcribe using OpenAI API
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            }
            
            with open(temp_audio.name, 'rb') as f:
                files = {
                    'file': f,
                    'model': (None, 'gpt-4o-mini-transcribe'),
                    'response_format': (None, 'json')
                }
                
                response = requests.post(
                    "https://api.openai.com/v1/audio/transcriptions",
                    headers=headers,
                    files=files,
                    timeout=30
                )
            
            # Clean up temp file
            os.unlink(temp_audio.name)
            
            if response.status_code == 200:
                data = response.json()
                return jsonify({
                    "success": True,
                    "text": data["text"]
                })
            else:
                return jsonify({
                    "success": False,
                    "error": f"Transcription failed: {response.text}"
                }), 500
                
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Transcription error: {str(e)}"
        }), 500

@app.route('/api/generate-speech', methods=['POST'])
def generate_speech():
    """Convert text to speech using OpenAI TTS"""
    try:
        validate_api_keys()
    except ValueError as e:
        return jsonify({"error": str(e)}), 500
    
    data = request.get_json()
    text = data.get('text', '')
    voice = data.get('voice', 'alloy')  # Default to alloy voice
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    if len(text) > 4096:
        return jsonify({"error": "Text too long (max 4096 characters)"}), 400
    
    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "tts-1",
            "input": text,
            "voice": voice,
            "response_format": "mp3"
        }
        
        response = requests.post(
            "https://api.openai.com/v1/audio/speech",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            # Return audio as base64 for easy frontend handling
            import base64
            audio_b64 = base64.b64encode(response.content).decode('utf-8')
            
            return jsonify({
                "success": True,
                "audio": audio_b64,
                "format": "mp3"
            })
        else:
            return jsonify({
                "success": False,
                "error": f"TTS failed: {response.text}"
            }), 500
            
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"TTS error: {str(e)}"
        }), 500

@app.route('/api/voice-session', methods=['POST'])
def voice_session():
    """Handle voice-based Fapulous session - combines transcription, AI response, and TTS"""
    try:
        validate_api_keys()
    except ValueError as e:
        return jsonify({"error": str(e)}), 500
    
    # Get form data
    step = request.form.get('step')
    session_data = json.loads(request.form.get('session_data', '{}'))
    voice = request.form.get('voice', 'alloy')
    
    # Handle audio file if present
    user_message = ""
    if 'audio' in request.files:
        audio_file = request.files['audio']
        
        if audio_file.filename != '':
            # Transcribe audio first
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_audio:
                    audio_file.save(temp_audio.name)
                    
                    headers = {
                        "Authorization": f"Bearer {OPENAI_API_KEY}"
                    }
                    
                    with open(temp_audio.name, 'rb') as f:
                        files = {
                            'file': f,
                            'model': (None, 'gpt-4o-mini-transcribe'),
                            'response_format': (None, 'json')
                        }
                        
                        transcribe_response = requests.post(
                            "https://api.openai.com/v1/audio/transcriptions",
                            headers=headers,
                            files=files,
                            timeout=30
                        )
                    
                    os.unlink(temp_audio.name)
                    
                    if transcribe_response.status_code == 200:
                        transcribe_data = transcribe_response.json()
                        user_message = transcribe_data["text"]
                    else:
                        return jsonify({"error": "Transcription failed"}), 500
                        
            except Exception as e:
                return jsonify({"error": f"Transcription error: {str(e)}"}), 500
    else:
        # Fallback to text message
        user_message = request.form.get('user_message', '')
    
    # Process AI response (reuse existing logic)
    if step == 'start':
        prompt = FAPULOUS_PROMPTS["SYSTEM_PROMPT"] + "\n\n" + FAPULOUS_PROMPTS["QUESTION_1_PROMPT"]
        user_content = "The user just masturbated and wants guidance."
    elif step == 'question1':
        prompt = FAPULOUS_PROMPTS["SYSTEM_PROMPT"] + "\n\n" + FAPULOUS_PROMPTS["QUESTION_2_PROMPT"]
        user_content = f"User's first response: {user_message}"
    elif step == 'question2':
        prompt = FAPULOUS_PROMPTS["SYSTEM_PROMPT"] + "\n\n" + FAPULOUS_PROMPTS["QUESTION_3_PROMPT"]
        user_content = f"User's second response: {user_message}"
    else:
        return jsonify({"error": "Invalid step"}), 400
    
    # Get AI text response
    def call_openai_chat(system_prompt, user_content):
        try:
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                "temperature": 0.7,
                "max_tokens": 100
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                assistant_message = data["choices"][0]["message"]["content"]
                return {
                    "success": True,
                    "response": assistant_message.strip(),
                    "error": None
                }
            else:
                return {
                    "success": False,
                    "response": None,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "response": None,
                "error": str(e)
            }
    
    ai_result = call_openai_chat(prompt, user_content)
    
    if not ai_result["success"]:
        return jsonify({"error": ai_result["error"]}), 500
    
    ai_response_text = ai_result["response"]
    
    # Convert AI response to speech
    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        tts_payload = {
            "model": "tts-1",
            "input": ai_response_text,
            "voice": voice,
            "response_format": "mp3"
        }
        
        tts_response = requests.post(
            "https://api.openai.com/v1/audio/speech",
            headers=headers,
            json=tts_payload,
            timeout=30
        )
        
        if tts_response.status_code == 200:
            import base64
            audio_b64 = base64.b64encode(tts_response.content).decode('utf-8')
            
            # Update session data
            if user_message:
                if 'conversation' not in session_data:
                    session_data['conversation'] = []
                session_data['conversation'].append({
                    "user": user_message,
                    "ai": ai_response_text,
                    "step": step
                })
            
            return jsonify({
                "success": True,
                "ai_response": ai_response_text,
                "user_message": user_message,
                "audio": audio_b64,
                "format": "mp3",
                "step": step,
                "next_step": get_next_step(step),
                "session_data": session_data
            })
        else:
            return jsonify({"error": "TTS generation failed"}), 500
            
    except Exception as e:
        return jsonify({"error": f"TTS error: {str(e)}"}), 500

# For Vercel deployment - this needs to be accessible
application = app

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=3002) 