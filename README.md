# ⚡ BotTrainer — LLM-Based NLU Studio

A production-grade Natural Language Understanding (NLU) system powered by **LLaMA 3.3** via Groq. BotTrainer classifies user intents and extracts entities in real time using prompt engineering and few-shot learning — no traditional ML training required.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)
![Groq](https://img.shields.io/badge/Groq-LLaMA3.3-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🧠 What It Does

| Feature | Description |
|---|---|
| Intent Classification | Detects user intent from 10 categories |
| Entity Extraction | Pulls key details like location, date, food item |
| Confidence Scoring | Returns a 0–1 confidence score per prediction |
| Model Evaluation | Benchmarks accuracy across 30 labeled test samples |
| Interactive UI | Streamlit app with live analysis and dataset explorer |

---

## 🏗️ Architecture
```
User Message
     ↓
Prompt Engineering (Few-Shot)
     ↓
Groq API (LLaMA 3.3 70B)
     ↓
JSON Parser
     ↓
Intent + Entities + Confidence
```

---

## 📁 Project Structure
```
BOT_TRAINER/
├── data/
│   ├── intents.json          ← 10 intents, 100 training examples
│   └── eval_dataset.json     ← 30 labeled test samples
├── modules/
│   ├── llm_client.py         ← Groq API wrapper
│   ├── data_loader.py        ← JSON loader utilities
│   ├── intent_classifier.py  ← Intent classification module
│   ├── entity_extractor.py   ← Entity extraction module
│   └── nlu_pipeline.py       ← Core predict() pipeline
├── prompts/
│   ├── intent_prompt.txt     ← Few-shot intent prompt
│   └── entity_prompt.txt     ← Few-shot entity prompt
├── app/
│   └── main.py               ← Streamlit web application
├── evaluation/
│   └── evaluator.py          ← Accuracy, F1, confusion matrix
├── .env                      ← API keys (not pushed to GitHub)
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/MonijaV/Bot_Trainer.git
cd Bot_Trainer
```

### 2. Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Create virtual environment and install dependencies
```bash
uv venv
source .venv/bin/activate
uv add groq streamlit scikit-learn pandas matplotlib seaborn python-dotenv
```

### 4. Set up your API key
Create a `.env` file in the root folder:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get a free API key at [https://console.groq.com](https://console.groq.com)

### 5. Run the app
```bash
streamlit run app/main.py
```

---

## 🎯 Supported Intents

| Intent | Example |
|---|---|
| book_flight | "Book a flight to Delhi tomorrow" |
| order_food | "Order me a pizza" |
| check_weather | "What is the weather in Chennai?" |
| set_reminder | "Remind me at 9pm" |
| play_music | "Play some jazz music" |
| check_balance | "What is my bank balance?" |
| send_message | "Text John I will be late" |
| get_directions | "Navigate to the airport" |
| greet | "Hello there" |
| goodbye | "Bye" |

---

## 🛠️ Tech Stack

- **LLM:** LLaMA 3.3 70B via Groq Cloud
- **UI:** Streamlit
- **Language:** Python 3.11
- **Package Manager:** uv
- **Evaluation:** scikit-learn

---

## 📊 Sample Output
```json
{
  "user_text": "Book a flight to Delhi tomorrow",
  "intent": "book_flight",
  "confidence": 0.99,
  "entities": {
    "location": "Delhi",
    "date": "tomorrow"
  }
}
```

---
