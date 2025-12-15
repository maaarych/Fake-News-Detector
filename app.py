import os
import json
import re
import torch
import streamlit as st
import spacy

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, TypedDict

from spacy.cli import download as spacy_download

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END


# =====================
# DEVICE
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================
# LANGUAGE DETECTION
# =====================
def detect_lang(text: str) -> str:
    if re.search(r"[ІіЇїЄєҐґ]", text):
        return "uk"
    return "en"


# =====================
# SPACY NER
# =====================
@st.cache_resource
def load_spacy_model(name: str):
    try:
        return spacy.load(name)
    except OSError:
        spacy_download(name)
        return spacy.load(name)


nlp_tools = {
    "uk": load_spacy_model("uk_core_news_sm"),
    "en": load_spacy_model("en_core_web_sm"),
    "default": load_spacy_model("en_core_web_sm")
}


def extract_entities(text: str, lang: str):
    nlp = nlp_tools.get(lang, nlp_tools["default"])
    doc = nlp(text)
    return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]


def highlight_entities(text: str, entities):
    html = text
    for ent in sorted(entities, key=lambda x: len(x["text"]), reverse=True):
        html = html.replace(
            ent["text"],
            f"<mark><b>{ent['text']}</b> ({ent['label']})</mark>"
        )
    return html


# =====================
# ML MODEL
# =====================
class FakeNewsDetector:
    def __init__(self, model_dir: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
        self.model.eval()

    def predict(self, text: str):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        ).to(device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.softmax(logits, dim=-1)[0]
        label = int(torch.argmax(probs))
        confidence = float(probs[label])

        return {
            "label": "FAKE" if label == 1 else "TRUE",
            "confidence": confidence
        }


# =====================
# RAG TYPES
# =====================
class FactCheckState(TypedDict, total=False):
    claim: str
    urls: List[str]
    context_docs: List[Document]
    result: dict
    fake_probability: float


def format_docs(docs: List[Document]) -> str:
    out = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "")
        body = d.page_content[:1200]
        out.append(f"[Source {i}] {src}\n{body}")
    return "\n\n".join(out)


# =====================
# RAG PIPELINE
# =====================
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

search_en = DuckDuckGoSearchResults(max_results=5)
search_ua = DuckDuckGoSearchResults(max_results=5, region="ua-ua")


def search_web(state: FactCheckState):
    claim = state["claim"]
    if detect_lang(claim) == "uk":
        ua = search_ua.invoke(claim)[:3]
        en = search_en.invoke(claim)[:2]
        urls = [r["link"] for r in ua + en]
    else:
        urls = [r["link"] for r in search_en.invoke(claim)[:5]]

    return {**state, "urls": urls}


def build_context(state: FactCheckState):
    urls = [u for u in state["urls"] if u.startswith("http")]
    if not urls:
        return {**state, "context_docs": []}

    loader = WebBaseLoader(urls)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    store = InMemoryVectorStore(embeddings)
    store.add_documents(chunks)

    retrieved = store.similarity_search(state["claim"], k=8)
    return {**state, "context_docs": retrieved}


prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict fact-checker. "
     "Return JSON ONLY:\n"
     "{"
     '"truth_probability": float,'
     '"verdict": "TRUE" | "FALSE" | "UNCERTAIN",'
     '"explanation": string,'
     '"evidence_snippets": [string]'
     "}"),
    ("human", "CLAIM:\n{claim}\n\nCONTEXT:\n{context}")
])


def analyze_claim(state: FactCheckState):
    context = format_docs(state.get("context_docs", [])) or "No evidence found."
    msg = prompt.invoke({"claim": state["claim"], "context": context})
    parsed = json.loads(llm.invoke(msg).content)

    truth = float(parsed["truth_probability"])
    parsed["fake_probability"] = 1 - truth

    return {**state, "result": parsed, "fake_probability": parsed["fake_probability"]}


def build_graph():
    g = StateGraph(FactCheckState)
    g.add_node("search", search_web)
    g.add_node("context", build_context)
    g.add_node("analyze", analyze_claim)

    g.add_edge(START, "search")
    g.add_edge("search", "context")
    g.add_edge("context", "analyze")
    g.add_edge("analyze", END)

    return g.compile()


# =====================
# STREAMLIT UI
# =====================
st.set_page_config(page_title="Fake News Detection", layout="centered")
st.title("📰 Fake News Detection (ML + RAG + NER)")

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is not set")

text = st.text_area("Enter news text / headline")

if st.button("Analyze") and text.strip():
    lang = detect_lang(text)
    entities = extract_entities(text, lang)

    st.subheader("🧠 Named Entities (Explainability)")
    if entities:
        st.markdown(highlight_entities(text, entities), unsafe_allow_html=True)
        with st.expander("Raw entities"):
            st.json(entities)
    else:
        st.write("No entities detected.")

    with st.spinner("Running ML model..."):
        detector = FakeNewsDetector("./models/en")
        ml = detector.predict(text)

    st.subheader("🔎 ML Classification")
    st.write(f"Prediction: **{ml['label']}**")
    st.progress(ml["confidence"])
    st.write(f"Confidence: `{ml['confidence']:.2f}`")

    with st.spinner("Running RAG fact-checking..."):
        graph = build_graph()
        rag = graph.invoke({"claim": text})["result"]

    st.subheader("🌐 RAG Fact Check")
    st.write(f"Verdict: **{rag['verdict']}**")
    st.progress(rag["fake_probability"])
    st.write(f"Fake probability: `{rag['fake_probability']:.2f}`")
    st.write(rag["explanation"])

    with st.expander("Evidence"):
        for e in rag["evidence_snippets"]:
            st.markdown(f"- {e}")
