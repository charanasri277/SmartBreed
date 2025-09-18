# 🐄 Indian Cattle & Buffalo Breed Classifier 

An **AI-powered system** that predicts the breed of cattle or buffalo from an uploaded image.

Once the breed is identified, the system provides:

- 📖 A detailed **breed summary** (via Groq LLM)  
- 🔊 **Audio narration** in multiple languages for farmers  
- ▶️ **YouTube care videos** for better livestock management  

---

## 🏆 Hackathon Journey

This project was created during the **Smart India Hackathon (SIH) Inhouse Journey** under the problem statement:

- **Domain:** Husbandry & Dairying  
- **Title:** Image-based breed recognition for cattle and buffaloes of India  
- **Problem ID:** SIH25004  
- **Category:** Software  
- **Theme:** Agriculture, FoodTech & Rural Development  

✨ **Achievement:** Our team received an **Excellence Award 🏅** from the internal inhouse hackathon panelists, with strong positive feedback for **innovation, real-world impact, and farmer-friendly design.**

---

## 📑 Project Overview

India is home to many cattle & buffalo breeds, crucial for:

- 🍼 Dairy productivity  
- 🚜 Draught power  
- 🔄 Dual-purpose usage  

This project creates a **farmer-friendly smart assistant** that:

1. 📸 Predicts the breed from an uploaded photo  
2. 📖 Provides breed details (origin, features, milk yield, utility)  
3. 🔊 Offers audio narration in local languages  
4. ▶️ Suggests YouTube videos for care & management  

---

## 🚀 Features

✔️ Breed Prediction from uploaded images  
✔️ Breed Summaries powered by Groq LLM  
✔️ Audio Output in multiple farmer languages  
✔️ YouTube Care Video Suggestions  
✔️ Simple & Farmer-Friendly Web Interface  

---

## 📊 Dataset

- ~6000 images across **41 breeds** (e.g., Murrah, Sahiwal, Gir, Ongole, etc.)  
- Split: **80% training / 20% validation**  

**Challenges faced:**
- 📉 Imbalanced dataset (some breeds had <50 images)  
- 💡 Lighting, angle, and background variations  

---

## 🧠 Model Training Journey

### 🔹 Initial Attempt (Baseline CNN)
- Trained a simple CNN from scratch  
- Performance was **very low** due to limited & imbalanced data  

### 🔹 Improvements with Deep Learning
1. **Data Augmentation** → rotations, flips, zoom, shifts, brightness changes  
2. **Early Stopping** → stopped training when validation stopped improving  
3. **Transfer Learning (MobileNetV2)** → fine-tuned on 41 breeds  
4. **Dropout (0.5)** → reduced overfitting  
5. **ReduceLROnPlateau** → adjusted learning rate automatically  
6. **Model Checkpoint** → saved best-performing model  

---

## 🔍 How Classification Works

1. 📤 Farmer uploads an image (resized & normalized)  
2. 🤖 Model predicts breed probabilities (e.g., Murrah = 60%)  
3. ✅ The predicted breed is selected  
4. Once identified, the system provides:  
   - 📖 Breed summary (via Groq LLM)  
   - 🔊 Audio narration (gTTS / pyttsx3)  
   - ▶️ YouTube care videos  

---

## 🤖 Groq LLM Integration

- ✨ **Summarization:** Breed origin, traits, and utility  
- 🌐 **Multilingual Support:** English, Hindi, Tamil & more  
- 🔊 **Text-to-Speech:** Converts text into farmer-friendly audio  
- ▶️ **YouTube Fetcher:** Finds livestock care & feeding videos  

---

## 🛠️ Tech Stack

- **Deep Learning:** TensorFlow, Keras  
- **Model:** MobileNetV2 (Transfer Learning)  
- **Training:** EarlyStopping, ReduceLROnPlateau, ModelCheckpoint  
- **Data Augmentation:** Keras ImageDataGenerator  
- **LLM:** Groq (Summarization & Multilingual Support)  
- **TTS:** gTTS / pyttsx3  
- **UI:** Streamlit  

---

## 🚀 Future Improvements

- 🔹 Collect ≥1000 images per breed for balanced training  
- 🔹 Try EfficientNetV2 / Vision Transformers  
- 🔹 Use ensemble models for robustness  
- 🔹 Deploy on mobile (TensorFlow Lite) for offline usage  
- 🔹 Add confidence threshold → show top-3 predictions  

---

## 📲 Complete Workflow

1. 📤 Upload cattle/buffalo image  
2. 🤖 Model predicts breed  
3. 📖 Groq LLM generates summary  
4. 🔊 Audio narration in local language  
5. ▶️ YouTube care videos for guidance  

---

## ✅ Conclusion

This project is more than just a classifier — it is a **farmer-friendly AI assistant 🎯.**  
By combining **Deep Learning + LLM + Audio + YouTube**, it makes breed recognition **accessible, educational, and practical for rural India.**
