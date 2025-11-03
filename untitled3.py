import streamlit as st
import os
import tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Check and import dependencies with helpful error messages
try:
    import torch
    from transformers import pipeline
    from diffusers import StableDiffusionPipeline
except ImportError as e:
    st.error(f"Missing required library: {e}. Please install with: pip install torch transformers diffusers")
    st.stop()

try:
    import moviepy.editor as mp
except ImportError:
    st.error("Missing moviepy. Install with: pip install moviepy")
    st.stop()

try:
    import pysrt
except ImportError:
    st.error("Missing pysrt. Install with: pip install pysrt")
    st.stop()

try:
    from langdetect import detect
except ImportError:
    st.error("Missing langdetect. Install with: pip install langdetect")
    st.stop()

try:
    import spacy
except ImportError:
    st.error("Missing spacy. Install with: pip install spacy")
    st.stop()

# Load models (cached to avoid reloading)
@st.cache_resource
def load_models():
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        st.error("Spacy model not found. Please run: python -m spacy download en_core_web_sm")
        st.stop()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        pipe.to(device)
        pipe.enable_attention_slicing()
    except Exception as e:
        st.error(f"Failed to load Stable Diffusion model: {e}")
        st.stop()
    
    return nlp, pipe, device

def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"  # Fallback

def segment_text(text, lang, nlp):
    doc = nlp(text)
    segments = [sent.text for sent in doc.sents]
    return segments

def generate_image(prompt, pipe, style="cartoon style"):
    full_prompt = f"{prompt}, {style}"
    image = pipe(full_prompt, num_inference_steps=20).images[0]
    return image

def create_subtitles(segments, duration_per_segment=3):
    subs = pysrt.SubRipFile()
    start_time = 0
    for i, seg in enumerate(segments):
        end_time = start_time + duration_per_segment
        sub = pysrt.SubRipItem(
            index=i+1, 
            start=pysrt.SubRipTime(seconds=start_time), 
            end=pysrt.SubRipTime(seconds=end_time), 
            text=seg
        )
        subs.append(sub)
        start_time = end_time
    return subs

def create_video(images, subtitles, output_path, fps=1):
    clips = []
    for img in images:
        # Convert PIL Image to numpy array
        img_array = np.array(img)
        clip = mp.ImageClip(img_array).set_duration(3)  # 3 seconds per image
        clips.append(clip)
    
    video = mp.concatenate_videoclips(clips, method="compose")
    
    # Add subtitles overlay
    def add_text(get_frame, t):
        frame = get_frame(t)
        img = Image.fromarray(frame.astype('uint8'))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 40)
        except:
            font = ImageFont.load_default()
        
        current_sub = None
        for sub in subtitles:
            if sub.start.ordinal / 1000 <= t < sub.end.ordinal / 1000:
                current_sub = sub.text
                break
        
        if current_sub:
            # Add black background for better readability
            bbox = draw.textbbox((0, 0), current_sub, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (img.width - text_width) // 2
            y = img.height - 100
            draw.rectangle([x-10, y-10, x+text_width+10, y+text_height+10], fill="black")
            draw.text((x, y), current_sub, fill="white", font=font)
        
        return np.array(img)
    
    video = video.fl(add_text)
    video.write_videofile(output_path, fps=fps, codec="libx264", verbose=False, logger=None)

# Streamlit UI
st.title("🎬 Multilingual Text-to-Animated-Video Generator")
st.write("Paste your text (in any language), and generate a cartoon video with subtitles!")

# Add warning about resource requirements
st.warning("⚠️ This app requires significant computational resources and may take 5-15 minutes to generate a video. Consider using a GPU for faster processing.")

text_input = st.text_area(
    "Enter your text:", 
    height=200, 
    placeholder="Once upon a time, a little fox ran through the forest..."
)

if st.button("Generate Video"):
    if not text_input.strip():
        st.error("Please enter some text!")
    else:
        with st.spinner("Loading models... This may take a few minutes on first run."):
            nlp, pipe, device = load_models()
            st.success(f"Models loaded successfully! Using device: {device}")
        
        with st.spinner("Processing... This may take 5-15 minutes."):
            try:
                # Step 1: Detect language
                lang = detect_language(text_input)
                st.info(f"Detected language: {lang}")
                
                # Step 2: Segment text
                segments = segment_text(text_input, lang, nlp)
                st.write(f"Text segmented into {len(segments)} parts.")
                
                # Step 3: Generate images
                images = []
                progress_bar = st.progress(0)
                for i, seg in enumerate(segments):
                    st.write(f"Generating image {i+1}/{len(segments)}: {seg[:50]}...")
                    prompt = f"Scene: {seg}"
                    img = generate_image(prompt, pipe)
                    images.append(img)
                    progress_bar.progress((i + 1) / len(segments))
                
                st.success("All images generated!")
                
                # Step 4: Create subtitles
                subs = create_subtitles(segments)
                
                # Step 5: Create video in temp file
                st.write("Creating video with subtitles...")
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
                    output_path = temp_file.name
                create_video(images, subs, output_path)
                
                # Display video
                st.success("Video generated successfully! 🎉")
                st.video(output_path)
                
                # Download button
                with open(output_path, "rb") as f:
                    st.download_button(
                        "📥 Download Video", 
                        f, 
                        file_name="generated_video.mp4", 
                        mime="video/mp4"
                    )
                
                # Cleanup
                try:
                    os.unlink(output_path)
                except:
                    pass
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.write("**Troubleshooting tips:**")
                st.write("- Try shorter text (2-3 sentences)")
                st.write("- Ensure all dependencies are installed")
                st.write("- Check that you have enough disk space and memory")

st.write("---")
st.write("**Tips:**")
st.write("- Keep text short (2-3 sentences) for faster generation")
st.write("- First run will be slower as models are downloaded")
st.write("- GPU acceleration highly recommended")
st.write("- Each sentence generates one image (3 seconds of video)")
