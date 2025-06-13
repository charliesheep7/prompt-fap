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

# Default conversation configuration
DEFAULT_CONVERSATION_CONFIG = {
    "rounds": 5,
    "system_prompt": """You are Fapulous-AI, a warm, evidence-based coach focused on men's post-orgasm recovery to help him achieve his goal: Feel less guilty after masturbation. Your tone is supportive, concise, and lightly humorous when appropriate. Cite the science in plain words once, **without academic jargon or hyperlinks**. Never moralise or shame; assume masturbation is normal.""",
    "round_prompts": [
        "Acknowledge the user's current mood and normalize it using neuroscience. Be supportive and validating. Keep it under 30 words.",
        "Provide a gentle technique or insight based on their mood. Be encouraging and mention progress. Keep it under 30 words.", 
        "Offer a practical tip or coping strategy. Be warm and understanding. Keep it under 30 words.",
        "Give reassurance and perspective on their experience. Be empathetic and hopeful. Keep it under 30 words.",
        "End with a personalized affirmation that starts with 'I...' based on their mood and responses. Keep it under 25 words."
    ]
}

# Mood options for user selection
MOOD_OPTIONS = [
    {"id": "stressed", "label": "Stressed", "emoji": "😰"},
    {"id": "guilty", "label": "Guilty", "emoji": "😔"},
    {"id": "relieved", "label": "Relieved", "emoji": "😌"},
    {"id": "zen", "label": "Zen", "emoji": "🧘"},
    {"id": "anxious", "label": "Anxious", "emoji": "😟"},
    {"id": "ashamed", "label": "Ashamed", "emoji": "😳"},
    {"id": "confused", "label": "Confused", "emoji": "😵"},
    {"id": "peaceful", "label": "Peaceful", "emoji": "😇"},
    {"id": "frustrated", "label": "Frustrated", "emoji": "😤"},
    {"id": "neutral", "label": "Neutral", "emoji": "😐"}
]

# Configure clients
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current conversation configuration and mood options"""
    return jsonify({
        "conversation_config": DEFAULT_CONVERSATION_CONFIG,
        "mood_options": MOOD_OPTIONS
    })

@app.route('/api/config', methods=['POST'])
def update_config():
    """Update conversation configuration"""
    data = request.get_json()
    
    # Update rounds if provided
    if 'rounds' in data:
        rounds = data['rounds']
        if not isinstance(rounds, int) or rounds < 3 or rounds > 10:
            return jsonify({"success": False, "error": "Rounds must be between 3 and 10"}), 400
        DEFAULT_CONVERSATION_CONFIG['rounds'] = rounds
        
        # Adjust round_prompts array to match new rounds count
        current_prompts = DEFAULT_CONVERSATION_CONFIG['round_prompts']
        if len(current_prompts) < rounds:
            # Add default prompts if we need more
            default_prompt = "Continue the supportive conversation. Be warm and encouraging. Keep it under 30 words."
            while len(current_prompts) < rounds:
                current_prompts.append(default_prompt)
        elif len(current_prompts) > rounds:
            # Trim if we have too many
            DEFAULT_CONVERSATION_CONFIG['round_prompts'] = current_prompts[:rounds]
    
    # Update system prompt if provided
    if 'system_prompt' in data:
        DEFAULT_CONVERSATION_CONFIG['system_prompt'] = data['system_prompt']
    
    # Update round prompts if provided
    if 'round_prompts' in data:
        if len(data['round_prompts']) != DEFAULT_CONVERSATION_CONFIG['rounds']:
            return jsonify({"success": False, "error": "Number of round prompts must match rounds count"}), 400
        DEFAULT_CONVERSATION_CONFIG['round_prompts'] = data['round_prompts']
    
    return jsonify({"success": True, "config": DEFAULT_CONVERSATION_CONFIG})

@app.route('/api/fapulous-session', methods=['POST'])
def fapulous_session():
    """Handle the Fapulous guided meditation session"""
    # Validate API keys are present
    try:
        validate_api_keys()
    except ValueError as e:
        return jsonify({"error": str(e)}), 500
    
    data = request.get_json()
    current_round = data.get('current_round', 1)  # 1-based round number
    user_message = data.get('user_message', '')
    session_data = data.get('session_data', {})
    mood = session_data.get('selected_mood', 'neutral')
    
    # Get conversation config
    config = session_data.get('conversation_config', DEFAULT_CONVERSATION_CONFIG)
    total_rounds = config['rounds']
    
    # Validate round number
    if current_round < 1 or current_round > total_rounds:
        return jsonify({"error": f"Invalid round number. Must be between 1 and {total_rounds}"}), 400
    
    # Get the appropriate prompt for this round
    system_prompt = config['system_prompt']
    round_prompt = config['round_prompts'][current_round - 1]  # Convert to 0-based index
    
    # Build context based on mood and conversation history
    mood_context = f"The user is currently feeling {mood}."
    if current_round == 1:
        user_content = f"{mood_context} This is the first round of a {total_rounds}-round conversation. The user just masturbated and wants guidance."
    else:
        conversation_history = session_data.get('conversation', [])
        history_text = "\n".join([f"Round {i+1} - User: {msg['user']}, AI: {msg['ai']}" for i, msg in enumerate(conversation_history)])
        user_content = f"{mood_context}\n\nConversation history:\n{history_text}\n\nCurrent user message: {user_message}"
    
    # Create the full prompt
    full_prompt = f"{system_prompt}\n\nRound {current_round} instruction: {round_prompt}"
    
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
    result = call_openai_chat(full_prompt, user_content)
    
    if result["success"]:
        # Add conversation to session data
        if 'conversation' not in session_data:
            session_data['conversation'] = []
        
        # Add the current exchange
        if current_round > 1 or user_message:  # Don't add empty first round
            session_data['conversation'].append({
                "round": current_round,
                "user": user_message,
                "ai": result["response"]
            })
        
        # Determine if conversation is complete
        is_complete = current_round >= total_rounds
        next_round = current_round + 1 if not is_complete else None
        
        response_data = {
            "ai_response": result["response"],
            "current_round": current_round,
            "next_round": next_round,
            "total_rounds": total_rounds,
            "is_complete": is_complete,
            "session_data": session_data
        }
        
        return jsonify(response_data)
    else:
        return jsonify({"error": result["error"]}), 500

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
    
    # Get form data for conversation flow
    current_round = int(request.form.get('current_round', 1))
    mood = session_data.get('selected_mood', 'neutral')
    
    # Get conversation config
    config = session_data.get('conversation_config', DEFAULT_CONVERSATION_CONFIG)
    total_rounds = config['rounds']
    
    # Validate round number
    if current_round < 1 or current_round > total_rounds:
        return jsonify({"error": f"Invalid round number. Must be between 1 and {total_rounds}"}), 400
    
    # Get the appropriate prompt for this round
    system_prompt = config['system_prompt']
    round_prompt = config['round_prompts'][current_round - 1]
    
    # Build context based on mood and conversation history
    mood_context = f"The user is currently feeling {mood}."
    if current_round == 1:
        user_content = f"{mood_context} This is the first round of a {total_rounds}-round conversation. The user just masturbated and wants guidance."
    else:
        conversation_history = session_data.get('conversation', [])
        history_text = "\n".join([f"Round {i+1} - User: {msg['user']}, AI: {msg['ai']}" for i, msg in enumerate(conversation_history)])
        user_content = f"{mood_context}\n\nConversation history:\n{history_text}\n\nCurrent user message: {user_message}"
    
    # Create the full prompt
    full_prompt = f"{system_prompt}\n\nRound {current_round} instruction: {round_prompt}"
    
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
    
    ai_result = call_openai_chat(full_prompt, user_content)
    
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
            if 'conversation' not in session_data:
                session_data['conversation'] = []
            
            # Add the current exchange
            if current_round > 1 or user_message:  # Don't add empty first round
                session_data['conversation'].append({
                    "round": current_round,
                    "user": user_message,
                    "ai": ai_response_text
                })
            
            # Determine if conversation is complete
            is_complete = current_round >= total_rounds
            next_round = current_round + 1 if not is_complete else None
            
            return jsonify({
                "success": True,
                "ai_response": ai_response_text,
                "user_message": user_message,
                "audio": audio_b64,
                "format": "mp3",
                "current_round": current_round,
                "next_round": next_round,
                "total_rounds": total_rounds,
                "is_complete": is_complete,
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