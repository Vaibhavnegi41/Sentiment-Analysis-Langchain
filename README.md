# 🧠 Intelligent Sentiment Analysis
### *Uncover emotions, gain insights & receive actionable AI suggestions in seconds.*

> An AI-powered sentiment analysis web application that analyzes text, detects emotions, rates feedback, and provides smart strategic suggestions — all in real time using **LangChain**, **HuggingFace**, and **Streamlit**.

<br/>

---

##  Features

-  **Deep Sentiment Detection** — Classifies text as Positive, Negative, or Neutral with high accuracy
-  **Smart Rating System** — Rates feedback on a scale of 0 to 10
-  **Executive Summary** — Generates a concise one-line summary of the feedback
-  **AI Strategic Suggestions** — Provides tailored advice based on sentiment type:
  -  Positive → Enhancement & growth strategies
  -  Negative → Issue resolution & recovery advice
  -  Neutral → Balanced perspective & improvement tips
-  **Beautiful Dark UI** — Glassmorphism design with gradient colors and smooth animations
-  **Real-time Analysis** — Instant results powered by Qwen3 LLM via HuggingFace

<br/>

---

##  Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend** | Streamlit | Interactive Web UI |
| **LLM** | Qwen/Qwen3-Coder-Next | AI Text Analysis |
| **AI Framework** | LangChain | Chain & Prompt Management |
| **Model Hosting** | HuggingFace Endpoint | Cloud Model Inference |
| **Output Parsing** | Pydantic + PydanticOutputParser | Structured JSON Output |
| **Branching Logic** | RunnableBranch + RunnableLambda | Sentiment-based Routing |
| **Environment** | python-dotenv | Secure API Key Management |

<br/>

---

##  Architecture

```
User Input (Text Feedback)
         ↓
   Prompt Template 1
         ↓
   Qwen3 LLM Model
         ↓
  PydanticOutputParser
         ↓
  Structured Output:
  {
    summary: "...",
    sentiment: "positive/negative/neutral",
    rating: "0-10"
  }
         ↓
   RunnableBranch
  ┌──────┬──────┬──────┐
  ↓      ↓      ↓
Pos    Neg   Neutral
  ↓      ↓      ↓
Prompt2 Prompt3 Prompt4
  └──────┴──────┘
         ↓
   AI Suggestion
         ↓
   Streamlit UI Display
```

<br/>

---

##  Project Structure

```
sentiment-analysis/
│
├── app.py                  # Main Streamlit application
├── .env                    # API keys (never commit this)
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

<br/>

---

##  Getting Started

### Prerequisites

```bash
Python 3.10+
HuggingFace Account with API Token
```

### Installation

```bash
# Step 1 — Clone the repository
git clone https://github.com/Vaibhavnegi41/sentiment-analysis.git
cd sentiment-analysis

# Step 2 — Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Step 3 — Install dependencies
pip install -r requirements.txt

# Step 4 — Setup environment variables
# Create .env file and add:
HUGGINGFACEHUB_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx

# Step 5 — Run the application
streamlit run app.py
```

<br/>

---

## Requirements

```txt
streamlit
langchain
langchain-huggingface
langchain-core
pydantic
python-dotenv
```

Install all at once:
```bash
pip install streamlit langchain langchain-huggingface langchain-core pydantic python-dotenv
```

<br/>

---

##  Environment Setup

Create a `.env` file in the root directory:

```env
HUGGINGFACEHUB_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
```

Get your token:
```
1. Go to → https://huggingface.co
2. Profile → Settings → Access Tokens
3. New Token → Copy → Paste in .env
```

<br/>

---

## How It Works

### Step 1 — Input
User enters any text feedback, review, or statement in the text area.

### Step 2 — Sentiment Chain
```python
chain = prompt1 | model | pydantic_parser
```
The text is sent to **Qwen3 LLM** which returns structured JSON:
```json
{
  "summary": "User expresses strong dissatisfaction with service.",
  "sentiment": "negative",
  "rating": "2"
}
```

### Step 3 — Branch Chain
```python
RunnableBranch(
    (sentiment == "positive" → Enhancement prompt),
    (sentiment == "negative" → Recovery advice prompt),
    (sentiment == "neutral"  → Balanced suggestion prompt)
)
```
Based on detected sentiment, a **different AI prompt** is triggered automatically.

### Step 4 — Display
Results are displayed beautifully with color-coded cards, metrics, and suggestions.

<br/>

---

## Future Improvements

- Multi-language sentiment analysis support
- Emotion detection (happy, angry, sad, surprised)
- Batch analysis — analyze multiple feedbacks at once
- Export results as PDF or CSV report
- History tracking — save past analyses
- REST API using FastAPI for integration
- Fine-tuned model for domain-specific analysis

<br/>

---

## Contributing

Contributions are welcome! Feel free to:

```
1. Fork the repository
2. Create a new branch  →  git checkout -b feature/your-feature
3. Commit your changes  →  git commit -m "Added new feature"
4. Push to branch       →  git push origin feature/your-feature
5. Open a Pull Request
```

<br/>


---

<div align="center">

⭐ **If you found this project helpful, please give it a star on GitHub!** ⭐

*Built with ❤️ using LangChain + HuggingFace + Streamlit*

</div>
