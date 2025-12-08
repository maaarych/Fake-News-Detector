import json
import re
from typing import List, TypedDict

from dotenv import load_dotenv

load_dotenv()
import os

print(os.getenv("USER_AGENT"))

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END


def is_ukrainian(text: str) -> bool:
    """
    Simple heuristic: if text contains Ukrainian/Cyrillic letters,
    treat it as Ukrainian.
    """
    return bool(re.search(r"[А-Яа-яІіЇїЄєҐґ]", text))


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

search_tool_en = DuckDuckGoSearchResults(
    output_format="list",
    max_results=5,
    region="wt-wt",
)

search_tool_ua = DuckDuckGoSearchResults(
    output_format="list",
    max_results=5,
    region="ua-ua",
)


class FactCheckState(TypedDict, total=False):
    claim: str
    urls: List[str]
    context_docs: List[Document]
    raw_answer: str
    result: dict
    fake_probability: float


def format_docs(docs: List[Document]) -> str:
    """Format retrieved docs into a single context string for the LLM."""
    parts = []
    for i, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        title = meta.get("title", "").strip()
        source = meta.get("source", "").strip()
        header = f"[Source {i}] {title}\nURL: {source}\n"
        body = d.page_content.strip()
        if len(body) > 1500:
            body = body[:1500] + " [...]"
        parts.append(header + body)
    return "\n\n".join(parts)


def search_web(state: FactCheckState) -> FactCheckState:
    """
    Node 1:
    - If claim is Ukrainian -> get up to 3 UA + 2 EN sources.
    - Else -> get up to 5 EN sources.
    """
    claim = state["claim"]

    if is_ukrainian(claim):
        print("\n[search_web] Detected Ukrainian claim. Getting 3 UA + 2 EN sources...")

        ua_results = search_tool_ua.invoke(claim)
        en_results = search_tool_en.invoke(claim)

        ua_urls = [r["link"] for r in ua_results[:3]]
        en_urls = [r["link"] for r in en_results[:2]]

        urls = ua_urls + en_urls

        print("[search_web] UA URLs:")
        for u in ua_urls:
            print("  -", u)
        print("[search_web] EN URLs:")
        for u in en_urls:
            print("  -", u)
    else:
        print("\n[search_web] Non-Ukrainian (assume English) claim. Getting up to 5 EN sources...")

        en_results = search_tool_en.invoke(claim)
        urls = [r["link"] for r in en_results[:5]]

        print("[search_web] EN URLs:")
        for u in urls:
            print("  -", u)

    return {**state, "urls": urls}


def build_rag_context(state: FactCheckState) -> FactCheckState:
    """
    Node 2: load URLs, chunk, embed, build in-memory vector store,
    and retrieve most relevant chunks to the claim.
    """
    urls = state.get("urls", [])
    if not urls:
        raise ValueError("No URLs in state. search_web must run first.")

    valid_urls = [url for url in urls if url and url.strip() and url.startswith(('http://', 'https://'))]

    if not valid_urls:
        print("\n[build_rag_context] No valid URLs found. Skipping context retrieval.")
        return {**state, "context_docs": []}

    print(f"\n[build_rag_context] Valid URLs: {len(valid_urls)}/{len(urls)}")

    loader = WebBaseLoader(valid_urls)
    raw_docs: List[Document] = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(raw_docs)

    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(chunks)

    claim = state["claim"]
    retrieved_docs = vector_store.similarity_search(claim, k=10)

    print(f"\n[build_rag_context] Loaded {len(raw_docs)} pages, "
          f"{len(chunks)} chunks; retrieved {len(retrieved_docs)} chunks.")

    return {**state, "context_docs": retrieved_docs}


truth_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a strict fact-checking assistant.\n\n"
        "You receive:\n"
        "- A CLAIM (news text).\n"
        "- CONTEXT: a set of web evidence chunks.\n\n"
        "You must judge how likely it is that the CLAIM is TRUE.\n"
        "Important:\n"
        "- You are judging the CLAIM itself, not the general topic.\n"
        "- If evidence strongly supports the claim, it is probably TRUE.\n"
        "- If evidence strongly contradicts the claim, it is probably FALSE.\n"
        "- If evidence is weak/unclear/conflicting, be UNCERTAIN.\n"
        "- Respond in the SAME LANGUAGE as the CLAIM text.\n\n"
        "Output must be JSON ONLY, matching this schema exactly:\n"
        "{{\n"
        '  "truth_probability": float,        // in [0,1], 0=definitely false, 1=definitely true,\n'
        '  "verdict": "TRUE" | "FALSE" | "UNCERTAIN",\n'
        '  "explanation": "short explanation focused on the CLAIM",\n'
        '  "evidence_snippets": ["snippet1", "snippet2"]\n'
        "}}\n\n"
        "Guidelines for truth_probability:\n"
        "- 0.0–0.2: strong evidence the CLAIM is false.\n"
        "- 0.2–0.4: probably false but some doubts.\n"
        "- 0.4–0.6: uncertain / not enough evidence.\n"
        "- 0.6–0.8: probably true.\n"
        "- 0.8–1.0: strong evidence the CLAIM is true.\n"
    ),
    (
        "human",
        "CLAIM:\n{claim}\n\n"
        "CONTEXT (web evidence chunks):\n{context}\n"
    )
])


def analyze_claim(state: FactCheckState) -> FactCheckState:
    """
    Node 3: LLM analyzes evidence and outputs truth_probability in [0,1],
    then we convert it into fake_probability = 1 - truth_probability.
    """
    docs = state.get("context_docs", [])
    if not docs:
        print("\n[analyze_claim] WARNING: No context docs available. Using claim only.")
        context_str = "No web evidence available."
    else:
        context_str = format_docs(docs)

    messages = truth_prompt.invoke({
        "claim": state["claim"],
        "context": context_str
    })

    resp = llm.invoke(messages)
    answer_text = resp.content

    print("\n[analyze_claim] Raw LLM output:")
    print(answer_text)

    try:
        parsed = json.loads(answer_text)
    except json.JSONDecodeError:
        start = answer_text.find("{")
        end = answer_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(answer_text[start: end + 1])
            except json.JSONDecodeError:
                parsed = {}
        else:
            parsed = {}

    truth_prob = None
    if isinstance(parsed, dict) and "truth_probability" in parsed:
        try:
            truth_prob = float(parsed["truth_probability"])
        except (TypeError, ValueError):
            truth_prob = None

    if truth_prob is None:
        verdict = (parsed.get("verdict") or "").upper()
        if verdict == "TRUE":
            truth_prob = 0.9
        elif verdict == "FALSE":
            truth_prob = 0.1
        else:
            truth_prob = 0.5

    parsed.setdefault("truth_probability", truth_prob)
    fake_prob = 1.0 - truth_prob
    parsed["fake_probability"] = fake_prob

    return {
        **state,
        "raw_answer": answer_text,
        "result": parsed,
        "fake_probability": fake_prob,
    }


def build_graph():
    workflow = StateGraph(FactCheckState)

    workflow.add_node("search_web", search_web)
    workflow.add_node("build_rag_context", build_rag_context)
    workflow.add_node("analyze_claim", analyze_claim)

    workflow.add_edge(START, "search_web")
    workflow.add_edge("search_web", "build_rag_context")
    workflow.add_edge("build_rag_context", "analyze_claim")
    workflow.add_edge("analyze_claim", END)

    return workflow.compile()


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY is not set. Export it before running.\n")

    graph = build_graph()

    claim_text = input("Enter news text / claim: ").strip()
    if not claim_text:
        print("Empty claim, exiting.")
        raise SystemExit

    final_state: FactCheckState = graph.invoke({"claim": claim_text})

    print("\n=== RAG FAKENESS RESULT ===")
    print("URLs used:")
    for u in final_state.get("urls", []):
        print("  -", u)

    print("\nParsed JSON result:")
    print(json.dumps(final_state.get("result", {}), indent=2, ensure_ascii=False))

    print("\nFake probability (0=real, 1=fake):", final_state.get("fake_probability"))
