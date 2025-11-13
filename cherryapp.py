# Import necessary libraries
import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import re
import nltk
import matplotlib.pyplot as plt
import random
import json
import io
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from streamlit.components.v1 import iframe  # For embedding external content
import time
import psutil  # For memory usage tracking

# Configure Streamlit app UI
st.set_page_config(
    page_title="🍒 Cherry: Mental Health Chatbot",
    page_icon="🍒",
    layout="centered"
)

# Display a motivational welcome quote
st.markdown("""🌸 *“Be kind to yourself. You're doing the best you can.”*""")

# Custom CSS styling for background and text
st.markdown(
    """
    <style>
        body {
            background-color: #fffbe6;
        }
        .stApp {
            background-color: #fffbe6;
        }
        h1, h2, h3 {
            color: #b4004e;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Download necessary NLTK models
nltk.download('vader_lexicon')
nltk.download('wordnet')


# Load the trained model with Streamlit caching
@st.cache_resource
def load_model_cached():
    return tf.keras.models.load_model("lstm_depression_model_with_phq.keras")


model = load_model_cached()

# Load tokenizer and NLP tools
with open("tokenizer.pickle", "rb") as f:
    tokenizer = pickle.load(f)

lemmatizer = WordNetLemmatizer()
vader = SentimentIntensityAnalyzer()
max_length = 300


# Function to clean and lemmatize user text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return " ".join([lemmatizer.lemmatize(w) for w in text.split()])


# Pad input sequences to a consistent length
def custom_pad_sequences(sequences, maxlen):
    return tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=maxlen, padding='post')


# Generate a prediction from the model
def predict_with_inputs(text_input, sentiment_input, phq_input):
    try:
        prediction = model.predict({
            "text_input": text_input,
            "sentiment_input": sentiment_input,
            "phq_input": phq_input
        })
        return prediction[0][0]
    except Exception:
        st.error("Oops! There was an error during prediction. Using a fallback response.")
        return 0.4


# Interpret PHQ-8 score into a depression category
def get_phq_label(score):
    if score <= 4:
        return "Minimal or No Depression"
    elif score <= 9:
        return "Mild Depression"
    elif score <= 14:
        return "Moderate Depression"
    else:
        return "Severe Depression"


# Display comparison of different model inputs
def show_bar_chart(full, text_only, sentiment_only, phq_only):
    labels = ['Full Input', 'Text Only', 'Sentiment Only', 'PHQ-8 Only']
    scores = [full, text_only, sentiment_only, phq_only]
    colors = ['#4CAF50', '#2196F3', '#FFC107', '#F44336']
    fig, ax = plt.subplots()
    bars = ax.bar(labels, scores, color=colors)
    ax.set_ylim(0, 1)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f"{yval:.2f}", ha='center')
    st.pyplot(fig)


# Track user state between interactions
if "name" not in st.session_state:
    st.session_state.name = ""
if "started" not in st.session_state:
    st.session_state.started = False
if "ready_for_phq" not in st.session_state:
    st.session_state.ready_for_phq = False
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False
if "analysis_completed" not in st.session_state:
    st.session_state.analysis_completed = False
if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False


# Handle user's name input
def set_name_and_start():
    name = st.session_state.get("name_input", "")
    if name:
        lowered = name.lower()
        if any(x in lowered for x in ["don't", "dont", "no name", "prefer not", "skip"]):
            st.session_state.name = "there"
            st.session_state.chat_log.append(("user", "I prefer not to share my name"))
            st.session_state.chat_log.append(("cherry", "That's totally okay! I'm here for you no matter what. 💛"))
        else:
            st.session_state.name = name
            st.session_state.chat_log.append(("user", name))
            st.session_state.chat_log.append(
                ("cherry", f"Nice to meet you, {name}! I'm Cherry 🍒, your mental health companion."))
        st.session_state.started = True


# Set flags to control UI
def set_ready_for_phq_true():
    st.session_state.ready_for_phq = True


def set_ready_for_phq_false():
    st.session_state.chat_log.append(("cherry", "No problem! I'm still here if you ever want to talk more. 💬"))
    st.session_state.prediction_done = False


# Handle user chat input
def handle_chat_input():
    user_text = st.session_state.chat_input
    if user_text:
        st.session_state.chat_log.append(("user", user_text))
        cleaned = clean_text(user_text)
        sentiment = vader.polarity_scores(cleaned)["compound"]
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = custom_pad_sequences(seq, max_length)
        dummy_phq = np.zeros((1, 8))

        # Track time and memory used for prediction
        process = psutil.Process()
        start_time = time.time()
        start_mem = process.memory_info().rss / (1024 ** 2)

        prediction = predict_with_inputs(padded, np.array([[sentiment]]), dummy_phq)

        end_time = time.time()
        end_mem = process.memory_info().rss / (1024 ** 2)

        prediction_time = end_time - start_time
        memory_used = end_mem - start_mem

        st.info(f"⏱️ Prediction Time: {prediction_time:.4f} sec")
        st.info(f"🧠 Memory Used During Prediction: {memory_used:.2f} MB")

        # Generate empathetic chatbot response based on prediction
        negative_responses = [
            "Thanks for sharing. It sounds like you're facing some challenges. Remember, you're not alone. 💛",
            "Thank you for sharing your feelings. I'm here to support you and together we can take small steps forward. 💚"
        ]
        positive_responses = [
            "I'm glad you're feeling okay! Remember, I'm always here if you want to chat more. 🌟",
            "Great to hear that things are okay, but I'm still here if you ever need someone to talk to. 💫"
        ]

        response = random.choice(negative_responses if prediction >= 0.5 else positive_responses)
        st.session_state.chat_log.append(("cherry", response))

        # Invite user to take PHQ-8
        st.session_state.chat_log.append(("cherry",
                                          "Sometimes it's hard to put our feelings into words, so I use a short questionnaire called **PHQ-8**. "
                                          "It helps us better understand how you're really feeling, based on your recent experiences.\n\n"
                                          "It only takes a minute, and your answers will stay private — just between us. 💬\n\n"
                                          "Would you like to continue?"
                                          ))

        st.session_state.prediction_done = True


# Render main interface
st.title("🍒 Meet Cherry: Your Mental Health Chatbot")

# Introduction screen
if not st.session_state.started:
    st.markdown("""
    #### 🌿 Welcome to Cherry, your mental wellness companion.

    This chatbot is here to gently support you and help detect early signs of depression  
    — through conversation, self-reflection, and a short, optional questionnaire (PHQ-8). 

    ⚠️ Please note: Cherry is not a diagnostic tool and does not provide medical advice. 
                    For professional help, always consult a licensed mental health provider. 

    You're never alone here. Let's take one small step together. 💗
    """)

# Collect user's name
if not st.session_state.started:
    st.text_input("👋 What's your name?", key="name_input", on_change=set_name_and_start)

# Chat interface
elif not st.session_state.ready_for_phq:
    for sender, message in st.session_state.chat_log:
        st.markdown(f"**{'You' if sender == 'user' else '🍒 Cherry'}:** {message}")

    if not st.session_state.prediction_done:
        st.text_input("💬 Tell me how you're feeling:", key="chat_input", on_change=handle_chat_input)

    elif st.session_state.prediction_done and not st.session_state.ready_for_phq:
        col1, col2 = st.columns(2)
        with col1:
            st.button("👍 Yes, I want to", key="ready_yes", on_click=set_ready_for_phq_true)
        with col2:
            st.button("🙅 No, maybe later", key="ready_no", on_click=set_ready_for_phq_false)

# PHQ-8 questionnaire section
if "phq_started" not in st.session_state:
    st.session_state.phq_started = False
if "phq_completed" not in st.session_state:
    st.session_state.phq_completed = False

# Display PHQ-8 form and analyze results
if st.session_state.ready_for_phq and not st.session_state.phq_completed:
    st.markdown("---")
    st.subheader("📋 PHQ-8 Questionnaire")
    phq_values = []
    questions = [
        "Little interest or pleasure in doing things?",
        "Feeling down, depressed, or hopeless?",
        "Trouble sleeping or sleeping too much?",
        "Feeling tired or having little energy?",
        "Poor appetite or overeating?",
        "Feeling bad about yourself?",
        "Trouble concentrating on things?",
        "Moving or speaking slowly or being fidgety?"
    ]
    for i, q in enumerate(questions):
        val = st.selectbox(q, [0, 1, 2, 3], key=f"phq_{i}")
        phq_values.append(val)

    if st.button("🧠 Analyze My Mental Health"):
        try:
            # Extract recent message and compute features
            cleaned = clean_text(st.session_state.chat_log[-3][1])
            sentiment = vader.polarity_scores(cleaned)["compound"]
            seq = tokenizer.texts_to_sequences([cleaned])
            padded = custom_pad_sequences(seq, max_length)
            sentiment_input = np.array([[sentiment]])
            phq_input = np.array([phq_values])

            # Make predictions with different input combinations
            full = predict_with_inputs(padded, sentiment_input, phq_input)
            dummy_text = np.zeros((1, max_length))
            neutral_sentiment = np.array([[0.0]])
            zero_phq = np.zeros((1, 8))
            text_only = predict_with_inputs(padded, neutral_sentiment, zero_phq)
            sentiment_only = predict_with_inputs(dummy_text, sentiment_input, zero_phq)
            phq_only = predict_with_inputs(dummy_text, neutral_sentiment, phq_input)

            # Show results and suggestions
            label = "Depressed" if full >= 0.5 else "Not Depressed"
            phq_score = sum(phq_values)

            st.subheader("📊 Results")
            st.write(f"**PHQ-8 Total Score:** {phq_score} / 24")
            st.write(f"**Interpretation:** {get_phq_label(phq_score)}")
            st.write(f"**Sentiment Score:** {sentiment:.4f}")
            st.write(f"**Model Prediction:** {full:.4f} → {label}")

            # Show conditional warnings or advice
            if phq_score <= 3 and full >= 0.6:
                st.warning("⚠️ You mentioned feeling down, but your PHQ-8 score is low. Stay mindful. 💡")
            elif phq_score >= 15 and full < 0.4:
                st.warning("⚠️ Your PHQ-8 score shows concern. Please reach out to someone you trust.")
            elif phq_score <= 3 and full <= 0.4:
                st.success("🌞 You're doing great! Keep taking care of yourself. 💚")
            elif 5 <= phq_score <= 9:
                st.warning("🌱 You’re showing mild signs of depression. Nothing to panic about.")
                if st.button("💡 Want gentle wellbeing suggestions?"):
                    st.markdown("- Talk to a friend 💬\n- Go for a short walk 🚶\n- Listen to uplifting music 🎶")
            elif 10 <= phq_score < 15 and full >= 0.5:
                st.warning("💛 You seem to be experiencing moderate depression — you might need some support.")
                st.markdown("""
                - ✍️ **Journaling:** Write your thoughts for 10 minutes  
                - 🌿 **Nature:** Take a short walk outside or open a window for fresh air  
                - 🧘 **Mindfulness:** Try a 5-minute breathing exercise  
                - 📚 **Resources:** Check [NHS self-help](https://www.nhs.uk/mental-health/self-help/)
                """)
            elif phq_score >= 15 and full >= 0.6:
                st.error("🚨 Both your message and PHQ-8 responses suggest severe depression. You're not alone ❤️")
                st.markdown(
                    "Talking to a professional might really help. Consider reaching out to a mental health helpline.")
                st.markdown(
                    "**📞 Need urgent support?**  \nCall **[Samaritans UK](https://www.samaritans.org)** at **116 123** *(free, 24/7)*")

            # Embed music recommendations
            st.markdown("### 🎵 Here's today's music to lift your mood:")
            st.components.v1.iframe("https://open.spotify.com/embed/track/33trZRsRCHDPemnACBcLJJ", height=80)

            st.markdown("### 🧘 Or a relaxing music track:")
            st.components.v1.iframe("https://open.spotify.com/embed/track/0Mw9vFnsi69Ipor7g7uJcO", height=80)

            st.session_state.phq_completed = True
            st.session_state.analysis_completed = True

        except Exception:
            st.error("An error occurred during analysis. Please try again later.")

# Feedback section after analysis
if st.session_state.analysis_completed:
    st.markdown("---")
    st.subheader("🙌 We Value Your Feedback")

    if not st.session_state.feedback_submitted:
        rating = st.radio("🌟 How helpful was this session?", ["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"])
        feedback = st.text_area("💬 Share any thoughts on your experience with Cherry:", key="user_feedback")

        if st.button("Submit Feedback", key="submit_feedback"):
            if feedback:
                if "feedback_log" not in st.session_state:
                    st.session_state.feedback_log = []
                st.session_state.feedback_log.append(feedback)
                st.session_state.feedback_submitted = True
            else:
                st.info("Please enter some feedback before submitting.")
    else:
        st.success("🎉 Thank you for your feedback!")
        quotes = [
            "🌞 Every day may not be good... but there's something good in every day.",
            "💛 You are stronger than you think.",
            "🌱 Healing isn’t linear — and that’s okay.",
            "☁️ Storms don’t last forever. Hang in there."
        ]
        st.info(random.choice(quotes))

# Chat log saving and download
st.markdown("---")
st.subheader("💾 Save or Download Your Chat")

if st.button("💾 Save Chat to File"):
    try:
        with open("my_chat_log.json", "w") as f:
            json.dump(st.session_state.chat_log, f)
        st.success("✅ Chat log saved as `my_chat_log.json`")
    except Exception as e:
        st.error(f"❌ Could not save chat: {e}")

buffer = io.StringIO()
json.dump(st.session_state.chat_log, buffer)
st.download_button(
    label="📤 Download Chat Log (JSON)",
    data=buffer.getvalue(),
    file_name="my_cherry_chat_log.json",
    mime="application/json"
)

# Restart session
st.markdown("---")
if st.button("🔁 Start Another Session"):
    st.session_state.clear()
    st.markdown("🔄 Restarting...")
    st.markdown('<meta http-equiv="refresh" content="0">', unsafe_allow_html=True)
