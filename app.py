from langgraph.graph import StateGraph,START,END
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict ,Annotated,Literal
from dotenv import load_dotenv
import os
from pydantic import BaseModel ,Field 
import operator
import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
    
    

# ----------------------------
# Load Environment
# ----------------------------
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# ----------------------------
# Initialize Model
# ----------------------------
# model = ChatGoogleGenerativeAI(
#     model="gemini-2.0-flash",
#     google_api_key=api_key
# )

model = ChatOllama(
    model="mistral",
    temperature=0
)


# ----------------------------
# Schemas
# ----------------------------
class SentimentSchema(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Sentiment of the review"
    )

class DiagnosisSchema(BaseModel):
    issue_type: Literal["UX", "Performance", "Bug", "Support", "Other"]
    tone: Literal["angry", "Frustrated", "Disappointed", "Calm"]
    urgency: Literal["low", "medium", "high"]

# Structured models
sentiment_model = model.with_structured_output(SentimentSchema)
diagnosis_model = model.with_structured_output(DiagnosisSchema)

# ----------------------------
# State Definition
# ----------------------------
class ReviewState(TypedDict):
    review: str
    sentiment: str
    diagnosis: dict
    response: str

# ----------------------------
# Nodes
# ----------------------------
def find_sentiment(state: ReviewState):
    prompt = f"Find the sentiment of this review:\n{state['review']}"
    result = sentiment_model.invoke(prompt)
    return {"sentiment": result.sentiment}

def check_sentiment(state: ReviewState):
    if state["sentiment"] == "positive":
        return "positive_response"
    else:
        return "run_diagnosis"

def positive_response(state: ReviewState):
    reply = f"Thank you for your positive feedback! 😊 We're glad you had a great experience."
    return {"response": reply}

def run_diagnosis(state: ReviewState):
    prompt = f"Diagnose the issue in this review:\n{state['review']}"
    result = diagnosis_model.invoke(prompt)
    return {"diagnosis": result.dict()}

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