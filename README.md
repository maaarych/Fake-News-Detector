# Fake News Detection (ML + RAG + NER)
HF space: https://huggingface.co/spaces/mmarych/FakeNewsDetection

This project combines:
- Transformer-based fake news classification
- RAG-based fact checking using live web evidence
- spaCy NER for explainability (EN / UK)

## Run locally
```bash
pip install -r requirements.txt
export OPENAI_API_KEY=your_key
streamlit run app.py


