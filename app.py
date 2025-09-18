import streamlit as st
import json
from PIL import Image
from phi.agent import Agent
from phi.model.groq import Groq
from utils import load_trained_model, preprocess_image, predict_breed
from youtube_search import fetch_youtube_links
import pyttsx3
import tempfile
import os
from gtts import gTTS
import io
import base64

# -----------------------------
# Streamlit Page Setup
# -----------------------------
st.set_page_config(page_title="Indian Cattle & Buffalo Breed Classifier", page_icon="🐄")
st.title("🐄 Indian Cattle & Buffalo Breed Classifier")
st.write("Upload an image of cattle or buffalo, and I'll classify its breed.")

# -----------------------------
# Language selection with audio support
# -----------------------------
languages = {
    "English": {"code": "en", "name": "English"},
    "Hindi": {"code": "hi", "name": "Hindi"},
    "Tamil": {"code": "ta", "name": "Tamil"},
    "Telugu": {"code": "te", "name": "Telugu"},
    "Bengali": {"code": "bn", "name": "Bengali"},
    "Marathi": {"code": "mr", "name": "Marathi"},
    "Gujarati": {"code": "gu", "name": "Gujarati"},
    "Kannada": {"code": "kn", "name": "Kannada"},
    "Malayalam": {"code": "ml", "name": "Malayalam"},
    "Punjabi": {"code": "pa", "name": "Punjabi"},
    "Urdu": {"code": "ur", "name": "Urdu"}
}
selected_lang = st.selectbox("🌐 Choose Language for Breed Summary", options=list(languages.keys()))

# -----------------------------
# Audio Generation Functions
# -----------------------------
def create_audio_gtts(text, language_code, slow=False):
    """Create audio using Google Text-to-Speech"""
    try:
        # Clean text for better audio output
        clean_text = text.replace("*", "").replace("-", "").replace("#", "")
        clean_text = " ".join(clean_text.split())  # Remove extra whitespace
        
        tts = gTTS(text=clean_text, lang=language_code, slow=slow)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tts.save(tmp_file.name)
            return tmp_file.name
    except Exception as e:
        st.error(f"Error creating audio: {str(e)}")
        return None

def create_audio_player(audio_file_path):
    """Create HTML audio player"""
    try:
        with open(audio_file_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
            audio_b64 = base64.b64encode(audio_bytes).decode()
            
        audio_html = f"""
        <audio controls style="width: 100%;">
            <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
            Your browser does not support the audio element.
        </audio>
        """
        return audio_html
    except Exception as e:
        st.error(f"Error creating audio player: {str(e)}")
        return None

def clean_up_temp_file(file_path):
    """Clean up temporary audio file"""
    try:
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)
    except:
        pass

# -----------------------------
# Load model & class indices
# -----------------------------
@st.cache_resource
def get_model():
    return load_trained_model("final_indian_bovine_breed_classifier_mobilenetv2.h5")

model = get_model()

with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# -----------------------------
# Setup Groq Agent
# -----------------------------
groq_agent = Agent(
    name="Breed Summary Agent",
    role="Expert on Indian cattle and buffalo breeds.",
    model=Groq(id="llama-3.3-70b-versatile"),
    instructions=[
        "You are an expert on Indian cattle and buffalo breeds.",
        "Provide concise, clear breed summaries with origin, appearance, uses, and unique features.",
        "Always format your response using proper markdown bullet points (- or *).",
        "Each bullet point should be on a new line and contain one key piece of information.",
        "Respond in the language specified by the user.",
        "Keep the content suitable for audio conversion - avoid complex formatting."
    ],
    markdown=True,
)

# -----------------------------
# File uploader
# -----------------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)
    
    # Preprocess & predict
    img_array = preprocess_image(img, target_size=(224, 224))
    breed, confidence, _ = predict_breed(model, img_array, class_indices)
    
    st.subheader("🔍 Prediction Result")
    st.success(f"**Predicted Breed:** {breed}")
    st.write(f"Confidence: **{confidence:.2f}%**")
    
    # -----------------------------
    # Breed Summary using Groq Agent
    # -----------------------------
    st.subheader("📘 Breed Summary")
    
    # Enhanced prompt with clearer formatting instructions
    prompt = (
        f"Provide a concise summary of the Indian cattle/buffalo breed '{breed}' in {languages[selected_lang]['name']}. "
        f"Format your response as markdown bullet points, with each point on a new line starting with '-' or '*'. "
        f"Include these aspects as separate bullet points:\n"
        f"- Origin and native region\n"
        f"- Physical appearance and characteristics\n"
        f"- Primary uses (milk, draft, meat, etc.)\n"
        f"- Unique features or special traits\n"
        f"Keep each bullet point concise (1-2 sentences max). Make the content suitable for audio narration."
    )
    
    with st.spinner("Generating breed summary..."):
        try:
            response = groq_agent.run(prompt)
            
            # Enhanced response processing
            if hasattr(response, 'content') and response.content:
                breed_summary = response.content
            elif hasattr(response, "output") and response.output:
                breed_summary = response.output
            elif hasattr(response, "outputs"):
                if isinstance(response.outputs, list) and response.outputs:
                    breed_summary = response.outputs[0]
                else:
                    breed_summary = str(response.outputs) if response.outputs else ""
            elif hasattr(response, "choices") and response.choices:
                breed_summary = response.choices[0].message.content
            else:
                breed_summary = str(response) if response else ""
            
            # Ensure bullet points are properly formatted
            if breed_summary:
                # Clean up the summary and ensure proper bullet point formatting
                lines = breed_summary.strip().split('\n')
                formatted_lines = []
                
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith(('-', '*', '•')):
                        # Add bullet point if missing
                        if line and not line.startswith('#'):  # Don't add bullets to headers
                            line = f"- {line}"
                    if line:  # Only add non-empty lines
                        formatted_lines.append(line)
                
                formatted_summary = '\n'.join(formatted_lines)
                st.markdown(formatted_summary)
                
                # -----------------------------
                # Audio Feature
                # -----------------------------
                st.subheader("🔊 Listen to Breed Summary")
                
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    if st.button("🎵 Generate Audio", type="primary"):
                        with st.spinner("Creating audio..."):
                            # Get language code for TTS
                            lang_code = languages[selected_lang]['code']
                            
                            # Create audio file
                            audio_file = create_audio_gtts(formatted_summary, lang_code)
                            
                            if audio_file:
                                # Store in session state for persistence
                                st.session_state.audio_file = audio_file
                                st.session_state.audio_generated = True
                                st.success("Audio generated successfully!")
                            else:
                                st.error("Failed to generate audio. Please try again.")
                
                with col2:
                    # Audio speed control
                    audio_speed = st.select_slider(
                        "🎛️ Audio Speed", 
                        options=["Slow", "Normal"], 
                        value="Normal",
                        help="Choose audio playback speed"
                    )
                
                # Display audio player if audio exists
                if hasattr(st.session_state, 'audio_generated') and st.session_state.audio_generated:
                    if hasattr(st.session_state, 'audio_file') and st.session_state.audio_file:
                        # Regenerate audio if speed changed
                        if st.button("🔄 Update Audio Speed"):
                            with st.spinner("Updating audio speed..."):
                                # Clean up old file
                                clean_up_temp_file(st.session_state.audio_file)
                                
                                lang_code = languages[selected_lang]['code']
                                slow_speed = (audio_speed == "Slow")
                                
                                new_audio_file = create_audio_gtts(formatted_summary, lang_code, slow=slow_speed)
                                if new_audio_file:
                                    st.session_state.audio_file = new_audio_file
                                    st.success("Audio speed updated!")
                        
                        # Create and display audio player
                        audio_html = create_audio_player(st.session_state.audio_file)
                        if audio_html:
                            st.markdown("### 🎧 Audio Player")
                            st.markdown(audio_html, unsafe_allow_html=True)
                            
                            # Download option
                            with open(st.session_state.audio_file, "rb") as audio_file:
                                st.download_button(
                                    label="📥 Download Audio",
                                    data=audio_file.read(),
                                    file_name=f"{breed}_summary_{selected_lang}.mp3",
                                    mime="audio/mpeg"
                                )
                        
                        # Cleanup button
                        if st.button("🗑️ Clear Audio"):
                            clean_up_temp_file(st.session_state.audio_file)
                            if hasattr(st.session_state, 'audio_file'):
                                del st.session_state.audio_file
                            if hasattr(st.session_state, 'audio_generated'):
                                del st.session_state.audio_generated
                            st.rerun()
                
                # Audio features info
                with st.expander("ℹ️ Audio Features"):
                    st.markdown("""
                    **🎵 Audio Features:**
                    - **Multi-language Support**: Audio generated in your selected language
                    - **Speed Control**: Choose between normal and slow playback
                    - **Download Option**: Save audio file for offline listening
                    - **Browser Compatible**: Works on all modern browsers
                    - **Accessibility**: Perfect for visually impaired users or hands-free learning
                    
                    **📱 Usage Tips:**
                    - Use headphones for better audio quality
                    - Slow speed is helpful for language learners
                    - Download audio for offline reference
                    """)
            else:
                st.warning("No breed summary generated. Please try again.")
                
        except Exception as e:
            st.error(f"Error generating breed summary: {str(e)}")
            # Fallback: Show a basic summary without AI
            st.info(f"Basic info: {breed} is an Indian cattle/buffalo breed. Please try refreshing for detailed information.")
    
    # -----------------------------
    # YouTube Video Suggestions
    # -----------------------------
    st.subheader("📺 Video Resources about this Breed")
    
    try:
        # Fetch YouTube links with multiple search queries for better coverage
        search_queries = [
            f"{breed} cattle breed",
            f"{breed} buffalo breed", 
            f"{breed} dairy farming",
            f"{breed} breed characteristics",
            f"Indian {breed} livestock",
            f"{breed} animal husbandry",
            f"{breed} farming techniques"
        ]
        
        all_links = []
        video_titles = []
        
        # Try different search terms to get at least 5 videos
        for query in search_queries:
            try:
                links = fetch_youtube_links(query)
                if links:
                    all_links.extend(links)
                    # If fetch_youtube_links returns titles too, add them
                    # Otherwise, we'll use generic descriptions
                    if len(all_links) >= 5:
                        break
            except:
                continue
        
        # Remove duplicates while preserving order
        unique_links = []
        seen = set()
        for link in all_links:
            if link not in seen:
                unique_links.append(link)
                seen.add(link)
        
        # Display at least 5 video suggestions
        if unique_links:
            # Create more descriptive titles for the videos
            video_descriptions = [
                f"🎥 Complete Guide to {breed} Breed",
                f"🐄 {breed} Characteristics and Features", 
                f"🥛 {breed} Dairy Farming Techniques",
                f"📚 Everything About {breed} Livestock",
                f"🎯 {breed} Breed Management Tips",
                f"🔍 {breed} vs Other Indian Breeds",
                f"💡 Modern {breed} Farming Methods",
                f"📖 Traditional {breed} Rearing Practices"
            ]
            
            # Display videos with descriptive titles
            for i, url in enumerate(unique_links[:8]):  # Show up to 8 videos
                if i < len(video_descriptions):
                    description = video_descriptions[i]
                else:
                    description = f"🎬 {breed} Related Video {i+1}"
                
                st.markdown(f"**{description}**")
                st.markdown(f"🔗 [Watch Video]({url})")
                st.markdown("---")
            
            # If we have fewer than 5 videos, add some generic search suggestions
            if len(unique_links) < 5:
                st.subheader("🔍 Additional Search Suggestions")
                additional_searches = [
                    f"Search: '{breed} cattle breed India'",
                    f"Search: '{breed} livestock farming'", 
                    f"Search: 'Indian {breed} dairy'",
                    f"Search: '{breed} animal characteristics'",
                    f"Search: '{breed} breeding techniques'"
                ]
                
                needed = 5 - len(unique_links)
                for i, search_term in enumerate(additional_searches[:needed]):
                    st.info(f"💡 {search_term}")
        else:
            # Fallback: Provide manual search suggestions
            st.info("📝 **Suggested YouTube Searches:**")
            search_suggestions = [
                f"🔍 '{breed} cattle breed characteristics'",
                f"🔍 '{breed} dairy farming in India'",
                f"🔍 '{breed} livestock management'", 
                f"🔍 'Indian {breed} breed documentary'",
                f"🔍 '{breed} animal husbandry techniques'"
            ]
            
            for suggestion in search_suggestions:
                st.markdown(f"• {suggestion}")
                
    except Exception as e:
        # Enhanced fallback with manual suggestions
        st.warning("Unable to fetch video links automatically. Here are some search suggestions:")
        
        manual_suggestions = [
            f"🎥 **'{breed} breed documentary'** - Learn about breed history and characteristics",
            f"🐄 **'{breed} cattle farming'** - Practical farming techniques and tips", 
            f"🥛 **'{breed} dairy production'** - Milk production and dairy management",
            f"📚 **'Indian {breed} livestock'** - Traditional and modern practices",
            f"🎯 **'{breed} animal care'** - Health, nutrition, and breeding tips"
        ]
        
        for suggestion in manual_suggestions:
            st.markdown(suggestion)
            st.markdown("---")

# -----------------------------
# Footer with Audio Info
# -----------------------------
st.markdown("---")
st.markdown("""
### 🔊 Audio Accessibility Features
This app supports audio narration in multiple Indian languages to make livestock information accessible to farmers with varying literacy levels. 
The audio feature helps bridge the digital divide in agricultural education.

**Supported Languages:** English, Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Urdu
""")

# Cleanup temporary files on app restart
if hasattr(st.session_state, 'audio_file'):
    try:
        # This will run when the session ends
        import atexit
        atexit.register(lambda: clean_up_temp_file(st.session_state.audio_file))
    except:
        pass
