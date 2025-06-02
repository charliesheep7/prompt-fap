# AI Model Comparison Chat Tool

A web application that allows you to have **multi-turn conversations** with three AI models side by side:
- **OpenAI GPT-4o-mini** 
- **Grok-2-1212 (X.AI)**
- **Gemini 2.0 Flash (Google)**

The tool is designed for **conversation comparison** with a specialized system prompt focused on reducing masturbation shame and promoting healthy attitudes.

## 📁 Required Files

Make sure you have received these files:

```
ai-chat-tool/
├── app.py                    # Main Flask application
├── templates/
│   └── index.html           # Web interface template  
├── requirements.txt         # Python dependencies
└── AI_CHAT_SETUP_README.md  # This instruction file
```

## 🔧 Prerequisites

- **Python 3.8+** installed on your system
- **pip** (Python package installer)
- **Internet connection** for API calls

## 📥 Installation Steps

### 1. Create Project Directory
```bash
mkdir ai-chat-tool
cd ai-chat-tool
```

### 2. Copy Files
Copy all the provided files into this directory:
- `app.py`
- `templates/index.html` (create `templates` folder first)
- `requirements.txt`

### 3. Create Virtual Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the Application
```bash
python app.py
```

The app will start and show:
```
* Running on http://localhost:3001
```

## 🌐 Access the Tool

1. Open your web browser
2. Navigate to: **http://localhost:3001**
3. Start chatting with the AI models!

## 🚀 How to Use

### Basic Usage
1. **Enter your message** in the text input field
2. **Click "Send"** or press Enter
3. **Compare responses** from all three AI models side by side
4. **Continue the conversation** - each model maintains context

### Advanced Features
- **Multi-turn conversations**: Each AI remembers the full conversation history
- **Clear Chat**: Reset all conversations with the "Clear Chat" button
- **Auto-scroll**: Conversations automatically scroll to show latest messages
- **Error handling**: Individual models show errors without affecting others

### Example Conversation Flow
```
1. You: "I feel guilty after masturbating"
2. AI responses appear side by side
3. You: "What specific techniques can help me overcome this guilt?"
4. AI models provide contextual follow-up responses
5. Continue the conversation...
```

## 🔑 API Keys

The API keys are **already hardcoded** in the `app.py` file:
- ✅ OpenAI API key included
- ✅ Grok (X.AI) API key included  
- ✅ Gemini (Google) API key included

**No additional setup required** for API access.

## 🛠️ System Prompt

The tool uses this system prompt for all models:
> "You're a mediator to help me reduce masturbation shame and lead me to better masturbation"

## 🐛 Troubleshooting

### Port Already in Use
If you see "Address already in use", either:
- Stop other applications using port 3001, or
- Change the port in `app.py` (last line): `app.run(debug=True, host='localhost', port=3002)`

### Flask Not Found
If you get "ModuleNotFoundError: No module named 'flask'":
```bash
# Make sure virtual environment is activated
source venv/bin/activate
pip install -r requirements.txt
```

### API Errors
- **OpenAI errors**: Check if the API key is valid and has usage quota
- **Grok errors**: Ensure X.AI API access is working
- **Gemini errors**: Verify Google AI API access

### Virtual Environment Issues
On some systems, you might need to use:
```bash
python -m venv venv
# instead of
python3 -m venv venv
```

## 🔄 Stopping the Application

To stop the app:
1. Go to the terminal where it's running
2. Press `Ctrl+C` (or `Cmd+C` on Mac)

To deactivate virtual environment:
```bash
deactivate
```

## 🎯 Features Overview

- ✅ **Three AI models** working simultaneously
- ✅ **Conversation history** maintained per model
- ✅ **Session persistence** during browser session
- ✅ **Responsive design** works on desktop and mobile
- ✅ **Real-time comparison** of AI responses
- ✅ **Error handling** for individual model failures
- ✅ **Modern UI** with loading states and animations

## 💡 Tips for Best Results

1. **Ask follow-up questions** to test how each model handles context
2. **Compare response styles** - some models are more clinical, others more empathetic
3. **Test edge cases** to see how each model handles sensitive topics
4. **Use "Clear Chat"** to start fresh conversations for different topics

## 📞 Support

If you encounter any issues:
1. Check this README for troubleshooting steps
2. Ensure all files are in the correct locations
3. Verify virtual environment is activated
4. Check terminal output for specific error messages

---

**Happy testing!** 🚀 This tool will help you compare how different AI models approach sensitive conversations and maintain context across multiple turns. 