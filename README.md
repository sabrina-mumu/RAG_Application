# 📚 Multilingual RAG System for HSC26 Bangla 1st Paper (LangChain + Gemini + FAISS)

This project builds a simple **Retrieval-Augmented Generation (RAG)** system capable of answering **Bangla and English** queries from a scanned Bangla textbook. It combines **OCR, chunking, semantic search**, and **generative AI (Google Gemini)** to answer natural language questions about the textbook.

---

## 🛠️ Setup Guide

### ✅ System Requirements (Colab/Linux)

```bash
sudo apt-get update
sudo apt-get install -y poppler-utils tesseract-ocr tesseract-ocr-ben
```

### ✅ Python Libraries

```bash
pip install langchain langchain_google_genai pytesseract pdf2image faiss-cpu sentence-transformers langdetect numpy
```

---

## 📄 Used Tools, Libraries & Packages

- **PDF to Image:** `pdf2image`
- **OCR Engine:** `Tesseract-OCR` (with Bangla support)
- **Text Embeddings:** `sentence-transformers` (multilingual model)
- **Similarity Search:** `FAISS`
- **RAG Framework:** `LangChain`
- **LLM:** `Google Gemini 1.5 Flash` via `langchain_google_genai`

---

## 📂 Input

Upload the provided PDF:  
**`HSC26-Bangla1st-Paper.pdf`**

Only pages **3 to 19** are processed to avoid noise.

---

## 💬 Sample Queries & Outputs

### ✅ Bangla Queries

| Query                                                | Answer                              |
|------------------------------------------------------|-------------------------------------|
| অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?              | উত্তর পাওয়া যায়নি ❌              |
| কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?       | অনুপমের মামাকে... ✅                |
| বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?              | উত্তর পাওয়া যায়নি ❌              |

### ✅ English Queries

| Query                                               | Answer                              |
|-----------------------------------------------------|-------------------------------------|
| Who is called the perfect man in Anupam's words?   | Anupam calls himself a "perfect man" ❌ |
| Who is referred to as Anupam's fate god?           | Answer not found ❌                 |

### ✅ Accuracy: `1 / 5 = 20%`

---

## ❓ Required Submission Questions

### 📌 1. What method or library did you use to extract the text, and why? Did you face any formatting challenges with the PDF content?

- **Used Tools:** `pdf2image` + `pytesseract` with `tesseract-ocr-ben`
- **Reason:** The textbook is a scanned image-based PDF. Tesseract supports Bangla OCR well, especially with the `tesseract-ocr-ben` package.
- **Challenges:** OCR from low-resolution scanned text led to:
  - Typos and artifacts
  - Sentence boundary confusion
  - Inconsistent spacing and alignment  
  These affected retrieval and answer accuracy.

---

### 📌 2. What chunking strategy did you choose? Why do you think it works well for semantic retrieval?

- **Strategy:** Sentence-based chunking with 2-sentence overlap
- **Reasoning:** In narrative/essay-style Bangla content, paragraph breaks are unreliable due to OCR errors. Sentence-level chunking maintains coherence while the overlap captures context for each chunk.

---

### 📌 3. What embedding model did you use? Why did you choose it? How does it capture the meaning of the text?

- **Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Why:** 
  - Supports over 50+ languages including Bangla and English
  - Efficient for Colab usage (small + accurate)
  - Captures sentence-level meaning and semantic closeness using transformer architecture
- **How:** It converts both document chunks and queries into dense vector representations that encode meaning beyond keyword matching.

---

### 📌 4. How are you comparing the query with your stored chunks? Why did you choose this similarity method and storage setup?

- **Comparison:** Cosine similarity between embedding vectors
- **Store:** `FAISS` vector database
- **Why:** 
  - Cosine similarity measures angular closeness of meanings
  - FAISS is optimized for fast and scalable dense vector search
- This allows quick retrieval of semantically similar text passages even with translation or paraphrasing.

---

### 📌 5. How do you ensure that the question and the document chunks are compared meaningfully? What would happen if the query is vague or missing context?

- **Mechanism:**
  - Both query and chunks are embedded using the same multilingual model
  - Overlapping chunks provide continuity
  - `langdetect` is used to detect the query language before passing it to the model
- **Limitations:**
  - If the query is vague, short, or ambiguous, retrieval becomes noisy
  - Missing context in OCR’d text (due to typos or chunk breaks) causes further errors

---

### 📌 6. Do the results seem relevant? If not, what might improve them?

- **Performance:** `20% accuracy` on test set suggests partial success
- **Issues Identified:**
  - OCR noise → missed key answers (e.g., “শুম্ভুনাথ” not matched)
  - Embedding model may miss Bangla literary nuances
  - Chunking may cut off answers mid-sentence

#### ✅ Potential Improvements:

- Clean OCR output using language models or regex post-processing
- Use larger or fine-tuned embedding models for Bangla (e.g., `LaBSE`)
- Add context-aware reranker (e.g., Gemini → rerank top-10 chunks)
- Use paragraph-based chunking with fallback to sentence-based when needed

---

## 📎 Conclusion

This RAG system provides a simple but extensible framework for **Bangla question-answering** using open-source tools and Gemini LLMs. With better OCR cleaning and stronger embeddings, it can scale to many subjects and languages.

---

## 📌 Authors

- Built by: **Sabrina Mostafij Mumu**
- Powered by: LangChain, Gemini, FAISS, Tesseract OCR
