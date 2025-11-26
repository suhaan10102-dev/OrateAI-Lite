import streamlit as st
import os
import time
import assemblyai as aai
import google.generativeai as genai
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from audio_recorder_streamlit import audio_recorder
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
from PIL import Image

try:
    config = st.secrets
    ASSEMBLY_API_KEY = config["api_keys"]["assemblyai"]
    GOOGLE_API_KEY = config["api_keys"]["google"]
    APP_USERNAME = config["auth"]["username"]
    APP_PASSWORD = config["auth"]["password"]
except Exception:
    st.error("Configuration error. Please check your secrets.toml file.")
    st.stop()

aai.settings.api_key = ASSEMBLY_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)

def check_authentication():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.title("Orate AI - Login")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            if submit:
                if username == APP_USERNAME and password == APP_PASSWORD:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        st.stop()

check_authentication()

st.set_page_config(page_title="Orate AI", page_icon="🎤", layout="wide")

if 'transcript_text' not in st.session_state:
    st.session_state.transcript_text = ""
if 'sentiment_results' not in st.session_state:
    st.session_state.sentiment_results = []
if 'filler_list' not in st.session_state:
    st.session_state.filler_list = []
if 'ai_feedback' not in st.session_state:
    st.session_state.ai_feedback = ""
if 'pdf_path' not in st.session_state:
    st.session_state.pdf_path = None
if 'wpm' not in st.session_state:
    st.session_state.wpm = 0
if 'posture_result' not in st.session_state:
    st.session_state.posture_result = None
if 'posture_image' not in st.session_state:
    st.session_state.posture_image = None

@st.cache_resource
def load_pose_model():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    return pose, mp_pose

@st.cache_resource
def load_posture_classifier():
    try:
        loaded = joblib.load('model.pkl')
        if isinstance(loaded, dict):
            model = loaded.get("model", None)
        else:
            model = loaded
        return model if model else None
    except Exception:
        return None

def calculate_wpm(transcript, audio_duration_ms):
    if not transcript:
        return 0
    word_count = len(transcript.split())
    duration_minutes = max((audio_duration_ms / 1000) / 60, 0.1)
    return round(word_count / duration_minutes)

def send_google(transcript_text, audience, language_style, feedback_length):
    model = genai.GenerativeModel(model_name="gemini-2.5-pro")
    prompt_text = f"Give professional public speaking feedback and tips to improve based on this speech. Also include an improved version of the speech. Give the speech in human tone. "
    if audience:
        prompt_text += f"Tailor the feedback and improved speech for a target audience of {audience}. "
    if language_style:
        prompt_text += f"Use a {language_style.lower()} language style appropriate for the audience.\n\n"
    if feedback_length:
        prompt_text += f"Use a {feedback_length.lower()} limit to give feedback.\n\n"
    prompt_text += transcript_text
    response = model.generate_content(prompt_text)
    return response.text

def pdf_export(transcript, sentiment_results, filler_data, ai_feedback, wpm):
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    filename = os.path.join(reports_dir, f"speech_report_{int(time.time())}.pdf")
    try:
        doc = SimpleDocTemplate(filename, pagesize=A4, leftMargin=inch, rightMargin=inch, 
                                topMargin=0.75*inch, bottomMargin=0.75*inch)
        story = []
        styles = getSampleStyleSheet()
        
        story.append(Paragraph("Speech Analysis Report", styles['Heading1']))
        story.append(Spacer(1, 0.25*inch))
        
        story.append(Paragraph("Speaking Rate:", styles['Heading2']))
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph(f"Words Per Minute (WPM): <b>{wpm}</b>", styles['Normal']))
        
        if wpm < 130:
            wpm_feedback = "Your speaking pace is slow. Consider speaking slightly faster."
        elif wpm > 170:
            wpm_feedback = "Your speaking pace is fast. Consider slowing down for clarity."
        else:
            wpm_feedback = "Your speaking pace is good (ideal range: 130-170 WPM)."
        
        story.append(Paragraph(wpm_feedback, styles['Normal']))
        story.append(Spacer(1, 0.25*inch))
        
        story.append(Paragraph("Transcribed Text:", styles['Heading2']))
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph(transcript or "(no transcript)", styles['Normal']))
        story.append(Spacer(1, 0.25*inch))
        
        story.append(Paragraph("Sentiment Analysis:", styles['Heading2']))
        story.append(Spacer(1, 0.1*inch))
        if sentiment_results:
            for result in sentiment_results:
                text_line = f'"{result.text}" → <b>{result.sentiment}</b> ({result.confidence*100:.1f}%)'
                story.append(Paragraph(text_line, styles['Normal']))
        else:
            story.append(Paragraph("No sentiment data available.", styles['Normal']))
        story.append(Spacer(1, 0.25*inch))
        
        story.append(Paragraph("Filler Words:", styles['Heading2']))
        story.append(Spacer(1, 0.1*inch))
        if filler_data:
            for f in filler_data:
                line = f"'{f['text']}' at {f['time']}s"
                story.append(Paragraph(line, styles['Normal']))
        else:
            story.append(Paragraph("No filler words detected.", styles['Normal']))
        story.append(Spacer(1, 0.25*inch))
        
        story.append(Paragraph("AI Feedback:", styles['Heading2']))
        story.append(Spacer(1, 0.1*inch))
        feedback_content = ai_feedback.replace('\n', '<br/>')
        story.append(Paragraph(feedback_content or "(no AI feedback)", styles['Normal']))
        
        doc.build(story)
        return filename
    except Exception as e:
        st.error(f"ReportLab PDF generation failed: {str(e)}")
        return None

def process_audio(audio_bytes, audience, language_style, feedback_length):
    audio_file = f"temp_audio_{int(time.time())}.wav"
    
    try:
        with open(audio_file, "wb") as f:
            f.write(audio_bytes)

        config = aai.TranscriptionConfig(
            speech_model=aai.SpeechModel.best,
            sentiment_analysis=True,
            word_boost=["um", "uh", "like", "you know"],
            boost_param="high"
        )

        with st.spinner("Transcribing audio..."):
            transcriber = aai.Transcriber(config=config)
            transcript = transcriber.transcribe(audio_file)

        if transcript.status == "error":
            st.error(f"Transcription error: {transcript.error}")
            return None

        wpm = calculate_wpm(transcript.text, transcript.audio_duration)
        st.session_state.wpm = wpm

        filler_list = []
        for word in transcript.words:
            if word.text.lower() in ["um", "uh", "like", "you know"]:
                time_sec = round(word.start / 1000, 2)
                filler_list.append({"text": word.text, "time": time_sec})

        with st.spinner("Getting AI feedback..."):
            ai_feedback = send_google(transcript.text, audience, language_style, feedback_length)

        st.session_state.transcript_text = transcript.text
        st.session_state.sentiment_results = transcript.sentiment_analysis
        st.session_state.filler_list = filler_list
        st.session_state.ai_feedback = ai_feedback

        with st.spinner("Generating PDF report..."):
            pdf_path = pdf_export(transcript.text, transcript.sentiment_analysis, filler_list, ai_feedback, wpm)
            if pdf_path:
                st.session_state.pdf_path = pdf_path

        return pdf_path
        
    finally:
        try:
            if os.path.exists(audio_file):
                time.sleep(0.5)
                os.remove(audio_file)
        except Exception as e:
            pass

def analyze_posture_from_image(image_file, pose, mp_pose, mp_drawing, classifier):
    try:
        image = Image.open(image_file)
        image_np = np.array(image)
        
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image_np
        
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        annotated_image = image_bgr.copy()
        posture_status = "No Pose Detected"
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )
            
            landmarks = results.pose_landmarks.landmark
            landmark_list = []
            for lm in landmarks:
                landmark_list.extend([lm.x, lm.y, lm.z])
            
            if classifier:
                try:
                    input_data = pd.DataFrame([landmark_list])
                    prediction = classifier.predict(input_data)
                    posture_status = str(prediction[0])
                except Exception as e:
                    posture_status = f"Prediction error: {str(e)}"
            else:
                posture_status = "Classifier not available"
            
            cv2.putText(annotated_image, posture_status, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(annotated_image, "No Pose Detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        return annotated_image_rgb, posture_status
        
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, f"Error: {str(e)}"

pose, mp_pose = load_pose_model()
mp_drawing = mp.solutions.drawing_utils
classifier = load_posture_classifier()

st.title("Orate AI")

with st.sidebar:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=250)

    st.markdown("---")
    st.subheader("Settings")
    audience = st.text_input("Target Audience", placeholder="e.g., Business professionals")
    language_style = st.selectbox("Language Style", ["Understandable", "Scientific"])
    feedback_length = st.selectbox("Feedback Length", ["Summarised(200 words)", "Detailled(1000 words)"])
    
    st.markdown("---")
    st.subheader("Quick Tips")
    tip_choice = st.radio("Select tip:", ["Posture", "Tone", "Confidence"], label_visibility="collapsed")
    if tip_choice == "Posture":
        st.info("Stand tall, keep your shoulders relaxed, and face your audience confidently.")
    elif tip_choice == "Tone":
        st.info("Vary your pitch and pace to keep your speech engaging.")
    else:
        st.info("Practice deep breathing before you speak to calm nerves.")

    st.markdown("---")

    if st.button("Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Audio & Analysis Controls")
    audio_bytes = audio_recorder(
        text="Click to record",
        recording_color="#e74c3c",
        neutral_color="#3498db",
        icon_size="2x",
    )

    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        
        if st.button("Analyze Speech", type="primary", use_container_width=True):
            try:
                pdf_path = process_audio(audio_bytes, audience, language_style, feedback_length)
                st.success("Analysis complete! Results loaded below.")
                if not pdf_path:
                    st.warning("PDF generation failed, but your results are displayed below.")
            except Exception as e:
                st.error(f"Analysis error: {str(e)}")
                st.info("Your results may still be visible below if transcription completed.")

with col2:
    st.subheader("Real-time Posture Detection")
    
    posture_image = st.camera_input("Capture your posture")
    
    if posture_image is not None:
        if 'last_posture_image' not in st.session_state or st.session_state.last_posture_image != posture_image:
            st.session_state.last_posture_image = posture_image
            with st.spinner("Analyzing posture..."):
                annotated_img, posture_result = analyze_posture_from_image(
                    posture_image, pose, mp_pose, mp_drawing, classifier
                )
                
                if annotated_img is not None:
                    st.session_state.posture_result = posture_result
                    st.session_state.posture_image = annotated_img
        
        if st.session_state.posture_result:
            if "Good" in st.session_state.posture_result or "good" in st.session_state.posture_result:
                st.success("**Perfect posture!**")
            elif "No Pose" in st.session_state.posture_result:
                st.info("Click 'Start' above to begin posture detection.")
            else:
                st.error("**Needs Improvement** - Keep your back and neck straight. Avoid bending or slouching.")

if st.session_state.transcript_text:
    st.markdown("---")
    st.subheader("Analysis Results")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Words Per Minute", st.session_state.wpm)
    with col2:
        st.metric("Word Count", len(st.session_state.transcript_text.split()))
    with col3:
        st.metric("Filler Words", len(st.session_state.filler_list))

    tab_transcript, tab_sentiment, tab_filler, tab_feedback = st.tabs(
        ["Transcript", "Sentiment", "Filler Words", "AI Feedback"]
    )

    with tab_transcript:
        st.text_area("Transcribed Text", st.session_state.transcript_text, height=300)

    with tab_sentiment:
        if st.session_state.sentiment_results:
            for result in st.session_state.sentiment_results:
                st.markdown(f'**"{result.text}"** → **{result.sentiment}** ({result.confidence*100:.1f}%)')
        else:
            st.info("No sentiment data available")

    with tab_filler:
        if st.session_state.filler_list:
            for word in st.session_state.filler_list:
                st.markdown(f"**'{word['text']}'** at **{word['time']}s**")
        else:
            st.success("No filler words detected! Great job!")

    with tab_feedback:
        st.markdown(st.session_state.ai_feedback)

    st.markdown("---")

    if st.session_state.pdf_path and os.path.exists(st.session_state.pdf_path):
        with open(st.session_state.pdf_path, "rb") as f:
            st.download_button(
                label="Download PDF Report",
                data=f,
                file_name=os.path.basename(st.session_state.pdf_path),
                mime="application/pdf",
                use_container_width=True,
                type="primary"
            )
    elif st.session_state.transcript_text:
        st.info("PDF generation encountered an issue, but you can copy the results above.")
