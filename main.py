import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from pydantic import BaseModel
from typing import Literal, Annotated
from langchain_core.runnables import RunnableLambda, RunnableBranch

load_dotenv()


st.set_page_config(page_title="Sentiment Analyzer", page_icon="✨", layout="centered")


st.markdown("""
<style>
    /* Global styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }
    
    /* Input Text Area Styles */
    .stTextArea textarea {
        background-color: rgba(30, 41, 59, 0.7) !important;
        color: #f8fafc !important;
        border: 1px solid #334155 !important;
        border-radius: 12px;
        padding: 16px;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .stTextArea textarea:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.3);
    }

    /* Primary Button */
    .stButton > button {
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 14px 0 rgba(37, 99, 235, 0.39) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px 0 rgba(37, 99, 235, 0.39) !important;
        background: linear-gradient(90deg, #60a5fa 0%, #3b82f6 100%) !important;
    }
    
    /* Secondary/Clear Button */
    [data-testid="column"]:nth-of-type(2) .stButton > button {
        background: linear-gradient(90deg, #475569 0%, #334155 100%) !important;
        box-shadow: 0 4px 14px 0 rgba(71, 85, 105, 0.39) !important;
    }
    [data-testid="column"]:nth-of-type(2) .stButton > button:hover {
        background: linear-gradient(90deg, #64748b 0%, #475569 100%) !important;
        box-shadow: 0 6px 20px 0 rgba(71, 85, 105, 0.39) !important;
    }

    /* Result Cards */
    .result-card {
        padding: 20px 24px;
        border-radius: 16px;
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 16px;
        font-size: 1.05rem;
        line-height: 1.6;
        color: #e2e8f0;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        transition: transform 0.3s ease;
    }
    .result-card:hover {
        transform: translateY(-2px);
    }
    
    /* Titles/Headers */
    .main-title {
        text-align: center;
        background: linear-gradient(to right, #60a5fa, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        margin-bottom: -5px !important;
    }
    .sub-title {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Metric styling adjustments */
    div[data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 800 !important;
        color: #f8fafc !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        color: #94a3b8 !important;
    }
</style>
""", unsafe_allow_html=True)


st.markdown("<h1 class='main-title'>Intelligent Sentiment Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Uncover emotions, gain insights & receive actionable AI suggestions in seconds.</p>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

language_model = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-Coder-Next",
    task="text-generation"
)

model = ChatHuggingFace(llm=language_model)

class Formatted(BaseModel):
    summary: str
    sentiment: Literal["positive", "negative", "neutral"]
    rating: Annotated[str, "rating from 0 to 10"]

pydantic_parser = PydanticOutputParser(pydantic_object=Formatted)
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="""
You are a strict JSON generator.

Analyze the sentiment of the text below.

Text:
{context}

{format_instructions}

Rules:
- Return ONLY valid JSON
- Do NOT return schema
- Do NOT explain anything
- ALL fields are REQUIRED

Output must contain:
- summary → short reason (1 line)
- sentiment → positive / negative / neutral
- rating → number from 0 to 10 as string

Example:
{{
  "summary": "The user expresses satisfaction and happiness.",
  "sentiment": "positive",
  "rating": "8"
}}
""",
    input_variables=["context"],
    partial_variables={
        "format_instructions": pydantic_parser.get_format_instructions()
    }
)

prompt2 = PromptTemplate(
    template="Give a suggestion or solution  improve or enhance performance of the services or products using this  feedback: {feedback}",
    input_variables=["feedback"]
)

prompt3 = PromptTemplate(
    template="Provide advice to handle this issue by using summary from the feedback : {feedback}",
    input_variables=["feedback"]
)

prompt4 = PromptTemplate(
    template="Give a neutral perspective or suggestion for this feedback: {feedback}",
    input_variables=["feedback"]
)

query = st.text_area("📝 Enter your statement", height=120, placeholder="Type your thoughts here...")

col1, col2 = st.columns(2)

with col1:
    analyze_btn = st.button("🔍 Analyze", use_container_width=True)

with col2:
    clear_btn = st.button("🧹 Clear", use_container_width=True)

if clear_btn:
    st.rerun()

def handlerFunction(x):
    return "Something went wrong!"

if analyze_btn:

    if query.strip() == "":
        st.warning("⚠️ Please enter a valid statement.")
    else:
        with st.spinner("Analyzing sentiment... 🤖"):

            chain = prompt1 | model | pydantic_parser
            structured_output = chain.invoke({"context": query})

            branch_chain = RunnableBranch(
                (lambda x: x.sentiment == "positive",
                 RunnableLambda(lambda x: x.summary) | prompt2 | model | parser),

                (lambda x: x.sentiment == "negative",
                 RunnableLambda(lambda x: x.summary) | prompt3 | model | parser),

                (lambda x: x.sentiment == "neutral",
                 RunnableLambda(lambda x: x.summary) | prompt4 | model | parser),

                RunnableLambda(handlerFunction)
            )

            suggestion = branch_chain.invoke(structured_output)

        st.success("✅ Analysis Complete!")

        st.divider()


        st.markdown("<h2 style='text-align: center; margin-bottom: 2rem;'>📊 Analysis Results</h2>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        sentiment_emojis = {
            "positive": "🟢 Positive",
            "negative": "🔴 Negative",
            "neutral": "🟡 Neutral"
        }

        col1.metric("Sentiment", sentiment_emojis.get(structured_output.sentiment, structured_output.sentiment.capitalize()))
        col2.metric("Rating Score", f"{structured_output.rating} / 10")

        st.markdown("<br>", unsafe_allow_html=True)

        border_colors = {
            "positive": "#10b981", 
            "negative": "#ef4444", 
            "neutral": "#eab308"   
        }
        b_color = border_colors.get(structured_output.sentiment, "#3b82f6")

        st.markdown("### 📝 Executive Summary")
        st.markdown(
            f"<div class='result-card' style='border-left: 5px solid {b_color};'>{structured_output.summary}</div>",
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("### 💡 AI Strategic Suggestion")
        st.markdown(
            f"<div class='result-card' style='border-left: 5px solid #a78bfa;'>{suggestion}</div>",
            unsafe_allow_html=True
        )