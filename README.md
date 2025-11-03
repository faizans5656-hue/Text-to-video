# 🎬 Multilingual Text-to-Animated Video Generator

This Streamlit app transforms any **text (in any language)** into a **cartoon-style animated video with subtitles** using AI models like **Stable Diffusion**, **SpaCy**, and **MoviePy**.  
Just enter your text, and the app will:
1. Detect the language  
2. Split text into segments  
3. Generate AI images for each scene  
4. Add subtitles  
5. Combine everything into a downloadable MP4 video  

---

## 🚀 Features

- 🌍 Multilingual text input (auto language detection)  
- 🧠 AI-based image generation using Stable Diffusion  
- 💬 Automatic subtitles synced with the video  
- 🎞️ Full video creation with overlayed text  
- ⚡ Streamlit-based simple web interface  

---

## 🧰 Technologies Used

- **Streamlit** – Web interface  
- **Transformers** & **Diffusers** – AI and Stable Diffusion pipeline  
- **SpaCy** – Natural language processing (sentence segmentation)  
- **LangDetect** – Language detection  
- **MoviePy** – Video generation and subtitle overlay  
- **Pillow (PIL)** – Image processing  
- **PySRT** – Subtitle creation  

---

## 📦 Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/text-to-video-generator.git
cd text-to-video-generator
