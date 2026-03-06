from langgraph.graph import StateGraph, START, END
from langchain_community.chat_models import ChatOllama
from typing import TypedDict, Literal
from dotenv import load_dotenv
import os
import streamlit as st
import json

# ----------------------------
# Load Environment
# ----------------------------
load_dotenv()

# ----------------------------
# Initialize Ollama Model
# ----------------------------
model = ChatOllama(
    model="mistral",
    temperature=0
)

# ----------------------------
# State Definition
# ----------------------------
class ReviewState(TypedDict, total=False):
    review: str
    sentiment: str
    diagnosis: dict
    response: str

# ----------------------------
# Nodes
# ----------------------------

def find_sentiment(state: ReviewState):
    prompt = f"""
    Classify the sentiment of this review as either
    "positive" or "negative".

    Return ONLY valid JSON like:
    {{"sentiment": "positive"}}

    Review:
    {state['review']}
    """

    result = model.invoke(prompt).content

    try:
        parsed = json.loads(result)
        return {"sentiment": parsed["sentiment"]}
    except:
        return {"sentiment": "negative"}  # fallback


def check_sentiment(state: ReviewState):
    if state["sentiment"] == "positive":
        return "positive_response"
    else:
        return "run_diagnosis"


def positive_response(state: ReviewState):
    reply = "Thank you for your positive feedback! 😊 We're glad you had a great experience."
    return {"response": reply}


def run_diagnosis(state: ReviewState):
    prompt = f"""
    Analyze the review and return JSON in this format:

    {{
        "issue_type": "UX | Performance | Bug | Support | Other",
        "tone": "angry | frustrated | disappointed | calm",
        "urgency": "low | medium | high"
    }}

    Review:
    {state['review']}

    Return ONLY JSON.
    """

    result = model.invoke(prompt).content

    try:
        parsed = json.loads(result)
        return {"diagnosis": parsed}
    except:
        # fallback safe values
        return {
            "diagnosis": {
                "issue_type": "Other",
                "tone": "disappointed",
                "urgency": "medium",
            }
        }


def negative_response(state: ReviewState):
    diagnosis = state["diagnosis"]

    reply = f"""
We’re sorry to hear about your experience.

Issue Type: {diagnosis['issue_type']}
Tone: {diagnosis['tone']}
Urgency: {diagnosis['urgency']}

Our team will work on resolving this as soon as possible.
"""
    return {"response": reply}


# ----------------------------
# Build Graph
# ----------------------------
graph = StateGraph(ReviewState)

graph.add_node("find_sentiment", find_sentiment)
graph.add_node("positive_response", positive_response)
graph.add_node("run_diagnosis", run_diagnosis)
graph.add_node("negative_response", negative_response)

graph.add_edge(START, "find_sentiment")
graph.add_conditional_edges("find_sentiment", check_sentiment)
graph.add_edge("positive_response", END)
graph.add_edge("run_diagnosis", "negative_response")
graph.add_edge("negative_response", END)

workflow = graph.compile()

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("📊 Review Analysis & Auto Response")

review_text = st.text_area("Enter a customer review:")

if st.button("Analyze Review"):
    if review_text.strip() == "":
        st.warning("Please enter a review.")
    else:
        initial_state = {"review": review_text}
        result = workflow.invoke(initial_state)

        st.subheader("Results")
        st.write("**Sentiment:**", result.get("sentiment"))

        if result.get("diagnosis"):
            st.write("**Diagnosis:**")
            st.json(result.get("diagnosis"))

        st.write("**Response:**")
        st.success(result.get("response"))