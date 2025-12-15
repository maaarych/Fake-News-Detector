import os
import json
import re
import time
import torch
import streamlit as st
import spacy
from typing import List, TypedDict

# Імпорти для моделі
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    XLMRobertaTokenizer, 
    XLMRobertaForSequenceClassification
)

# Імпорти для RAG
from spacy.cli import download as spacy_download
from langchain_openai import ChatOpenAI
# Використовуємо Tavily Client напряму (надійніше ніж LangChain tool)
from tavily import TavilyClient
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

# Налаштування сторінки
st.set_page_config(page_title="Hybrid Fake News Detector", layout="wide", page_icon="⚖️")

# =====================
# CONFIG
# =====================
MODEL_NAME = "mmarych/my-fake-news-roberta"
MODEL_SUBFOLDER = "model/en" 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hf_token = os.getenv("HF_TOKEN")
tavily_api_key = os.getenv("TAVILY_API_KEY")

def is_ukrainian(text: str) -> bool:
    return bool(re.search(r"[А-Яа-яІіЇїЄєҐґ]", text))

def clean_json_string(json_str: str) -> dict:
    clean_str = json_str.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(clean_str)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', clean_str, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                pass
        return {}

# =====================
# 1. NLP UTILS
# =====================
@st.cache_resource
def load_spacy_models():
    models = {}
    for lang, model_name in [("uk", "uk_core_news_sm"), ("en", "en_core_web_sm")]:
        try:
            models[lang] = spacy.load(model_name)
        except OSError:
            spacy_download(model_name)
            models[lang] = spacy.load(model_name)
    return models

nlp_models = load_spacy_models()

def highlight_entities(text: str, lang: str):
    nlp = nlp_models.get("uk" if is_ukrainian(text) else "en", nlp_models["en"])
    doc = nlp(text)
    html = text
    entities = {e.text: e.label_ for e in doc.ents}
    for text_ent, label in sorted(entities.items(), key=lambda x: len(x[0]), reverse=True):
        html = html.replace(
            text_ent,
            f"<mark style='background-color: #e0f2f1; border-radius: 4px; padding: 0 2px;'><b>{text_ent}</b> <span style='font-size: 0.7em; color: #666;'>{label}</span></mark>"
        )
    return html

# =====================
# 2. ML MODEL
# =====================
class FakeNewsDetector:
    def __init__(self, model_repo_id: str):
        try:
            self.tokenizer = XLMRobertaTokenizer.from_pretrained(
                model_repo_id, subfolder=MODEL_SUBFOLDER, token=hf_token
            )
            self.model = XLMRobertaForSequenceClassification.from_pretrained(
                model_repo_id, subfolder=MODEL_SUBFOLDER, token=hf_token
            ).to(device)
        except Exception:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_repo_id, subfolder=MODEL_SUBFOLDER, token=hf_token, use_fast=False
                )
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_repo_id, subfolder=MODEL_SUBFOLDER, token=hf_token
                ).to(device)
            except Exception as e_auto:
                raise RuntimeError(f"CRITICAL: Failed to load model. Error: {e_auto}")
        self.model.eval()

    def predict_proba(self, text: str) -> float:
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512, padding=True
        ).to(device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        return float(probs[1]) if len(probs) > 1 else float(probs[0])

@st.cache_resource(show_spinner=False)
def get_ml_detector():
    return FakeNewsDetector(MODEL_NAME)

# =====================
# 3. RAG PIPELINE (DIRECT TAVILY)
# =====================
class FactCheckState(TypedDict, total=False):
    claim: str
    search_context: str # ТУТ БУДЕ ТЕКСТ З ПОШУКУ
    urls: List[str]
    result: dict
    fake_probability: float
    debug_log: List[str]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

# Ініціалізація клієнта Tavily напряму
tavily_client = TavilyClient(api_key=tavily_api_key) if tavily_api_key else None

def robust_search(query: str, max_results=3):
    """Шукає і відразу повертає контент, без скрапінгу."""
    if not tavily_client:
        return [], ""
    
    try:
        # include_answer=True змушує Tavily дати коротку відповідь самому
        response = tavily_client.search(query, max_results=max_results, include_answer=True)
        results = response.get('results', [])
        
        # Збираємо текст прямо з результатів Tavily
        context_parts = []
        if response.get('answer'):
             context_parts.append(f"AI SUMMARY: {response['answer']}")
             
        for res in results:
            context_parts.append(f"SOURCE: {res['title']}\nURL: {res['url']}\nCONTENT: {res['content']}\n")
            
        return results, "\n\n".join(context_parts)
    except Exception as e:
        return [], str(e)

def search_web(state: FactCheckState) -> FactCheckState:
    claim = state["claim"]
    urls = []
    full_context = ""
    logs = []
    
    if not tavily_api_key:
        return {**state, "urls": [], "search_context": "No API Key"}

    try:
        if is_ukrainian(claim):
            logs.append("Mode: UA")
            # 1. Пошук по суті
            res1, ctx1 = robust_search(f"{claim}", max_results=3)
            # 2. Пошук спростування
            res2, ctx2 = robust_search(f"{claim} фейк правда", max_results=2)
            
            urls.extend([r['url'] for r in res1 + res2])
            full_context = ctx1 + "\n---\n" + ctx2
            
        else:
            logs.append("Mode: EN")
            # 1. Агресивний Fact-Check пошук (Це ловить Трампа)
            # Ми додаємо "fact check", "fake", "hoax" щоб знайти спростування
            queries = [
                f"{claim} fact check",
                f"is {claim} true or fake",
            ]
            
            for q in queries:
                res, ctx = robust_search(q, max_results=3)
                urls.extend([r['url'] for r in res])
                full_context += f"\nQuery: {q}\n{ctx}\n"
            
    except Exception as e:
        logs.append(f"Error: {e}")
        
    unique_urls = list(set([u for u in urls if u]))
    return {**state, "urls": unique_urls, "search_context": full_context, "debug_log": logs}

# Пропускаємо крок `build_rag_context`, бо ми вже маємо текст від Tavily
truth_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a strict fact-checking assistant.\n"
        "Analyze the provided SEARCH CONTEXT to verify the CLAIM.\n\n"
        "CRITICAL RULES:\n"
        "1. If the context contains words like 'fake', 'hoax', 'AI generated', 'false', 'debunked' in relation to the claim -> Verdict is FALSE.\n"
        "2. Even if you see the image described, check if the text says it is real or fake.\n"
        "3. If the context mentions Snopes, Reuters, AP Fact Check calling it false -> Verdict is FALSE.\n\n"
        "Respond in the SAME LANGUAGE as the CLAIM.\n\n"
        "Output JSON:\n"
        "{{\n"
        '  "truth_probability": float (0.0=FAKE, 1.0=TRUE),\n'
        '  "verdict": "TRUE" | "FALSE" | "UNCERTAIN",\n'
        '  "explanation": "concise proof",\n'
        '  "evidence_snippets": ["quote1", "quote2"]\n'
        "}}"
    ),
    (
        "human",
        "CLAIM:\n{claim}\n\n"
        "SEARCH CONTEXT:\n{context}\n"
    )
])

def analyze_claim(state: FactCheckState) -> FactCheckState:
    context = state.get("search_context", "")
    
    if not context or len(context) < 50:
        return {
            **state,
            "result": {"verdict": "UNCERTAIN", "explanation": "No data found.", "evidence_snippets": []},
            "fake_probability": 0.5
        }

    msg = truth_prompt.invoke({"claim": state["claim"], "context": context})
    resp = llm.invoke(msg)
    parsed = clean_json_string(resp.content)
    
    truth_prob = parsed.get("truth_probability")
    if truth_prob is None:
        v = parsed.get("verdict", "UNCERTAIN")
        truth_prob = 0.9 if v == "TRUE" else (0.1 if v == "FALSE" else 0.5)
    
    fake_prob = 1.0 - float(truth_prob)
    parsed["fake_probability"] = fake_prob
    return {**state, "result": parsed, "fake_probability": fake_prob}

def build_graph():
    g = StateGraph(FactCheckState)
    g.add_node("search_web", search_web)
    # Ми видалили build_rag_context, бо він не потрібен з Tavily
    g.add_node("analyze_claim", analyze_claim)
    
    g.add_edge(START, "search_web")
    g.add_edge("search_web", "analyze_claim")
    g.add_edge("analyze_claim", END)
    return g.compile()

def aggregate_scores(ml_prob: float, rag_prob: float, rag_verdict: str):
    if rag_verdict == "UNCERTAIN":
        return ml_prob, "ML Fallback", "RAG не знайшов доказів."
    
    # Якщо RAG каже FALSE (Fake > 0.8), ми віримо йому більше
    if rag_prob > 0.8:
         final_prob = (rag_prob * 0.8) + (ml_prob * 0.2)
         expl = "Факти вказують на фейк."
    else:
         final_prob = (rag_prob * 0.7) + (ml_prob * 0.3)
         expl = "Гібридна оцінка."
         
    return final_prob, "Hybrid", expl

# =====================
# UI
# =====================
st.title("⚖️ Hybrid Fake News Detector")

if not tavily_api_key:
    st.warning("⚠️ TAVILY_API_KEY не знайдено!")

text = st.text_area("Введіть текст новини:", height=150)

if st.button("🔍 Перевірити", type="primary") and text.strip():
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. ML Style Analysis")
        with st.status("Аналіз стилю...", expanded=True) as status:
            try:
                ml_detector = get_ml_detector() 
                ml_prob = ml_detector.predict_proba(text)
                status.update(label="Готово!", state="complete", expanded=False)
                
                ml_color = "red" if ml_prob > 0.5 else "green"
                st.markdown(f"Fake Probability (ML): <b style='color:{ml_color}'>{ml_prob:.2%}</b>", unsafe_allow_html=True)
                st.progress(ml_prob)
            except Exception as e:
                status.update(label="Помилка ML", state="error")
                ml_prob = 0.5 
            
    with col2:
        st.subheader("2. RAG Fact Check")
        with st.spinner("Перевірка фактів (Tavily)..."):
            try:
                rag_graph = build_graph()
                rag_res = rag_graph.invoke({"claim": text})
                rag_prob = rag_res.get("fake_probability", 0.5)
                
                rag_color = "red" if rag_prob > 0.5 else "green"
                st.markdown(f"Fake Probability (RAG): <b style='color:{rag_color}'>{rag_prob:.2%}</b>", unsafe_allow_html=True)
                st.progress(rag_prob)
            except Exception as e:
                st.error(f"RAG Error: {e}")
                rag_prob = 0.5
                rag_res = {"result": {"verdict": "ERROR"}}

    st.divider()
    res_dict = rag_res.get("result", {})
    final_score, method, expl = aggregate_scores(ml_prob, rag_prob, res_dict.get("verdict", "UNCERTAIN"))
    
    st.header(f"Final Verdict: {'FAKE' if final_score > 0.5 else 'REAL'} ({final_score:.2%})")
    
    with st.expander("Show Details"):
        st.write("RAG Verdict:", res_dict)
        st.write("Search Context (What LLM saw):")
        st.text(rag_res.get("search_context", "")[:1000] + "...")
        st.write("Sources:", rag_res.get("urls", []))
