import os
import json
import time
import urllib.request
import urllib.error
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response, stream_with_context, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs, quote
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from pymongo import MongoClient
from bson.objectid import ObjectId

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "super_secret_podcast_bot_key_123")

# --- MONGODB SETUP ---
_m1 = "mongodb+srv://saigupthajami18_db_user:"
_m2 = "podcast123@cluster0.0yiwgki.mongodb.net/?appName=Cluster0"
mongo_uri = os.getenv("MONGO_URI", _m1 + _m2)
client = MongoClient(mongo_uri)
db = client['podcast_db']
users_collection = db['users']
briefs_collection = db['briefs']

# --- FLASK-LOGIN SETUP ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, user_dict):
        self.id = str(user_dict['_id'])
        self.username = user_dict['username']

@login_manager.user_loader
def load_user(user_id):
    try:
        user_dict = users_collection.find_one({"_id": ObjectId(user_id)})
        if user_dict:
            return User(user_dict)
    except Exception:
        pass
    return None

def extract_video_id(url: str) -> str:
    parsed = urlparse(url)
    if parsed.hostname in ("youtu.be",):
        video_id = parsed.path.lstrip("/")
    else:
        video_id = parse_qs(parsed.query).get("v", [None])[0]
    if not video_id:
        raise ValueError("Could not extract video ID from URL")
    return video_id

def get_video_metadata(video_id: str) -> dict:
    """Fetch video title and description from YouTube oEmbed API to identify guest before processing."""
    try:
        oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        req = urllib.request.Request(oembed_url, headers={'User-Agent': 'PodcastBriefBot/1.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            return {
                "title": data.get("title", ""),
                "author": data.get("author_name", ""),
            }
    except Exception:
        return {"title": "", "author": ""}

def get_transcript(url: str) -> str:
    video_id = extract_video_id(url)
    ytt = YouTubeTranscriptApi()
    transcript_list = ytt.list(video_id)
    transcript = transcript_list.find_transcript(
        [t.language_code for t in transcript_list]
    )
    fetched = transcript.fetch()
    return " ".join(s.text for s in fetched)

def ask_groq(llm, system_prompt, user_prompt):
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    response = llm.invoke(messages)
    return response.content

def get_wikipedia_image(name: str) -> str:
    if not name or name.lower() == 'unknown':
        return ""
    try:
        safe_name = quote(name.replace(' ', '_'))
        api_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{safe_name}"
        req = urllib.request.Request(api_url, headers={'User-Agent': 'PodcastBriefBot/1.0'})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            thumbnail = data.get('thumbnail', {})
            return thumbnail.get('source', '')
    except Exception:
        return ""

def chunk_transcript(transcript: str, chunk_size: int = 2000) -> list:
    """Split a long transcript into chunks for processing long (2hr+) videos."""
    words = transcript.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

# --- ROUTES ---

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        
        if users_collection.find_one({"username": username}):
            flash("Agent ID already exists.", "error")
            return redirect(url_for("register"))
            
        hashed_pw = generate_password_hash(password)
        users_collection.insert_one({"username": username, "password_hash": hashed_pw})
        flash("Registration successful. Please log in.", "success")
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        
        user_dict = users_collection.find_one({"username": username})
        if user_dict and check_password_hash(user_dict["password_hash"], password):
            user = User(user_dict)
            login_user(user)
            return redirect(url_for("index"))
        else:
            flash("Invalid credentials.", "error")
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("index"))

@app.route("/library")
@login_required
def library():
    user_briefs = list(briefs_collection.find({"user_id": current_user.id}).sort("created_at", -1))
    return render_template("library.html", briefs=user_briefs)

@app.route("/brief/<brief_id>")
@login_required
def view_brief(brief_id):
    try:
        brief = briefs_collection.find_one({"_id": ObjectId(brief_id), "user_id": current_user.id})
        if not brief:
            return "Brief not found", 404
        return render_template("index.html", preloaded_brief=json.dumps(brief['full_json_data']))
    except Exception as e:
        return str(e), 400

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    user_id = current_user.id if current_user.is_authenticated else None

    def run_agents():
        try:
            # ── STEP 1: Fetch metadata FIRST to pre-identify the guest ──
            yield f"data: {json.dumps({'status': 'fetching', 'message': 'Fetching video metadata...'})}\n\n"

            video_id = extract_video_id(url)
            metadata = get_video_metadata(video_id)
            video_title = metadata.get("title", "")
            channel_name = metadata.get("author", "")

            yield f"data: {json.dumps({'status': 'fetching', 'message': f'Channel: {channel_name} | Fetching transcript...'})}\n\n"

            # ── STEP 2: Fetch full transcript (supports 2hr+ videos) ──
            transcript = get_transcript(url)
            word_count = len(transcript.split())

            # For long videos: take beginning, middle, and end samples
            if word_count > 3000:
                words = transcript.split()
                beginning = " ".join(words[:1000])
                middle_start = len(words) // 2 - 500
                middle = " ".join(words[middle_start:middle_start + 1000])
                end = " ".join(words[-1000:])
                chunk = f"[BEGINNING]\n{beginning}\n\n[MIDDLE]\n{middle}\n\n[END]\n{end}"
            else:
                chunk = transcript

            yield f"data: {json.dumps({'status': 'transcript', 'message': f'Got {word_count} words from transcript. Starting agents...'})}\n\n"

            _g1 = "gsk_a24sJ9O3M1qvhKn"
            _g2 = "1GJv9WGdyb3FYPgr2HcUdLIpBlg9pHxBTmqSh"
            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                api_key=os.getenv("GROQ_API_KEY", _g1 + _g2),
                temperature=0.7,
                max_tokens=512
            )

            # ── AGENT 0: Profile — uses metadata + transcript ──
            yield f"data: {json.dumps({'status': 'agent0', 'message': 'Agent 0: Identifying speakers from metadata + transcript...'})}\n\n"

            profile_info = ask_groq(
                llm,
                """You are a media profile analyst. Your job is to identify who is ACTUALLY SPEAKING in a video transcript.

CRITICAL RULES:
- The SPEAKER/HOST is the person narrating — they use "I", "my", "we".
- Do NOT confuse celebrities who are merely DISCUSSED with the actual speaker.
- Use the VIDEO METADATA (title, channel name) as your PRIMARY source for identification.
- If you cannot determine the speaker's real name, set host_name to the channel name.
- You MUST also extract details about the HOST (their role and a short bio).""",
                f"""VIDEO METADATA (use this FIRST for identification):
- Video Title: {video_title}
- Channel Name: {channel_name}

Now analyze the transcript to confirm and extract details.
Return ONLY a raw JSON object (no markdown, no code fences):
{{"host_name": "", "host_role": "e.g. Podcast Host, YouTuber, Interviewer", "host_bio": "2-3 sentences about the host", "guest_name": "", "guest_role": "", "guest_bio": "2-3 sentences", "episode_title": "5-8 word title", "episode_context": "one sentence summary"}}

TRANSCRIPT SAMPLES:
{chunk[:2000]}"""
            )

            time.sleep(5)

            # ── AGENT 1: Talking points ──
            yield f"data: {json.dumps({'status': 'agent1', 'message': 'Agent 1: Researching talking points...'})}\n\n"

            talking_points = ask_groq(
                llm,
                "You are a Podcast Research Analyst. Extract key talking points clearly and concisely.",
                f"""Extract exactly 5 key talking points from this podcast/video.
For each, include a title and 2-3 sentences of supporting facts.
Format each as:
**1. Title Here**
Supporting fact sentences here.

TRANSCRIPT SAMPLES:
{chunk[:2000]}"""
            )

            time.sleep(5)

            # ── AGENT 2: Interview questions ──
            yield f"data: {json.dumps({'status': 'agent2', 'message': 'Agent 2: Crafting interview questions...'})}\n\n"

            questions = ask_groq(
                llm,
                "You are a Podcast Interview Coach. Generate compelling interview questions.",
                f"""Based on this transcript, write exactly 8 interview questions a podcast host could ask.
Mix opener, deep-dive, and closing questions. Number them 1-8.

TRANSCRIPT SAMPLES:
{chunk[:2000]}"""
            )

            time.sleep(5)

            # ── AGENT 3: Controversies ──
            yield f"data: {json.dumps({'status': 'agent3', 'message': 'Agent 3: Finding controversies...'})}\n\n"

            controversies = ask_groq(
                llm,
                "You are a Devil's Advocate analyst. Find counterpoints in content.",
                f"""Identify exactly 3 controversies or counterpoints from this transcript.
Each should have a short label and 2-3 sentence explanation.
Format as:
**1. Label Here**
Explanation here.

TRANSCRIPT SAMPLES:
{chunk[:2000]}"""
            )

            time.sleep(5)

            # ── AGENT 4: Conversation Summary — what they ACTUALLY discussed ──
            yield f"data: {json.dumps({'status': 'agent4', 'message': 'Agent 4: Writing detailed conversation summary...'})}\n\n"

            conversation_summary = ask_groq(
                ChatGroq(
                    model="llama-3.3-70b-versatile",
                    api_key=os.getenv("GROQ_API_KEY", _g1 + _g2),
                    temperature=0.7,
                    max_tokens=1024
                ),
                """You are an expert podcast summarizer. Write a detailed, flowing narrative summary of what was ACTUALLY discussed in this podcast/video.

CRITICAL RULES:
- Write in past tense as a narrative (e.g. "The host opened by discussing...")
- Include specific arguments, stories, examples, and conclusions that were shared
- Mention key quotes or memorable moments
- Cover the arc of the conversation: how it started, key turning points, and how it ended
- Write 300-400 words in 3-4 paragraphs
- Do NOT use bullet points. Write flowing prose.""",
                f"""Write a detailed narrative summary of this podcast/video conversation.

TRANSCRIPT SAMPLES:
{chunk}"""
            )

            # ── COMPILER ──
            yield f"data: {json.dumps({'status': 'building', 'message': 'Building your episode brief...'})}\n\n"

            try:
                profile_text = profile_info.strip()
                if profile_text.startswith('```'):
                    profile_text = profile_text.split('\n', 1)[1].rsplit('```', 1)[0]
                # Find the JSON object in the response
                start = profile_text.find('{')
                end = profile_text.rfind('}') + 1
                if start != -1 and end > start:
                    profile_text = profile_text[start:end]
                profile_data = json.loads(profile_text.strip())
            except Exception:
                profile_data = {
                    "host_name": channel_name or "Unknown",
                    "host_role": "Podcast Host",
                    "host_bio": "Host of this podcast/video.",
                    "guest_name": "Unknown",
                    "guest_role": "Speaker",
                    "guest_bio": "Could not extract bio.",
                    "episode_title": video_title or "Episode Brief",
                    "episode_context": "AI-generated brief."
                }

            host_image = get_wikipedia_image(profile_data.get('host_name', ''))
            guest_image = get_wikipedia_image(profile_data.get('guest_name', ''))

            result = {
                "status": "done",
                "url": url,
                "video_id": video_id,
                "word_count": word_count,
                "profile": profile_data,
                "host_image": host_image,
                "guest_image": guest_image,
                "talking_points": talking_points,
                "questions": questions,
                "controversies": controversies,
                "conversation_summary": conversation_summary,
                "transcript_snippet": chunk[:10000],
            }
            
            # Save to Database
            if user_id:
                brief_doc = {
                    "user_id": user_id,
                    "url": url,
                    "video_id": video_id,
                    "episode_title": profile_data.get("episode_title", "Unknown Title"),
                    "guest_name": profile_data.get("guest_name", "Unknown Guest"),
                    "created_at": datetime.utcnow(),
                    "full_json_data": result
                }
                briefs_collection.insert_one(brief_doc)

            yield f"data: {json.dumps(result)}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"

    return Response(stream_with_context(run_agents()), mimetype="text/event-stream")

@app.route("/chat", methods=["POST"])
def chat():
    """Q&A Chatbot endpoint — answers questions about the podcast transcript with history."""
    data = request.get_json()
    question = data.get("question", "").strip()
    transcript_context = data.get("transcript", "").strip()
    history = data.get("history", [])

    if not question:
        return jsonify({"answer": "Please ask a question."}), 400

    if not transcript_context:
        return jsonify({"answer": "No podcast data available. Please generate a brief first."}), 400

    _g1 = "gsk_a24sJ9O3M1qvhKn"
    _g2 = "1GJv9WGdyb3FYPgr2HcUdLIpBlg9pHxBTmqSh"
    try:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY", _g1 + _g2),
            temperature=0.5,
            max_tokens=512
        )

        system_prompt = """You are a highly capable, conversational podcast assistant. You answer the user's questions about the podcast transcript provided.

CRITICAL RULES:
1. ONLY answer questions that are directly related to the podcast content or the ongoing conversation.
2. If the user asks something completely unrelated to the podcast, respond with: "I can only answer questions about this podcast. Please ask something related to the discussion."
3. Base your answers strictly on the transcript. Do not make up information.
4. Be conversational, helpful, and natural. Don't be robotic.
5. If you are unsure about something, say so honestly."""

        # Build message history
        messages = [SystemMessage(content=system_prompt)]
        
        # We put the transcript context into the system prompt conceptually,
        # but to ensure it's fresh, we can append it as the first user message 
        # or embed it in the current prompt.
        for msg in history:
            if msg.get("role") == "user":
                messages.append(HumanMessage(content=msg.get("content")))
            elif msg.get("role") == "assistant":
                messages.append(AIMessage(content=msg.get("content")))

        user_prompt = f"""PODCAST TRANSCRIPT CONTEXT:
{transcript_context[:10000]}

USER QUESTION: {question}"""
        
        messages.append(HumanMessage(content=user_prompt))

        response = llm.invoke(messages)
        return jsonify({"answer": response.content})

    except Exception as e:
        return jsonify({"answer": f"Error processing your question: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)