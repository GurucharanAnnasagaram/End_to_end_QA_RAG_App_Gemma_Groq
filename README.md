 Key Features
✅ PDF Processing – Loads and processes multiple PDFs using PyPDFDirectoryLoader.
✅ Vector Embeddings – Converts document text into embeddings using GoogleGenerativeAIEmbeddings.
✅ FAISS Vector Search – Enables efficient document retrieval via FAISS vector storage.
✅ RAG Pipeline – Uses retrieval-augmented generation to fetch relevant document context before generating answers.
✅ LLM Integration – Uses Groq's Gemma-2 9B IT model for question-answering.
✅ Streamlit UI – Allows users to upload PDFs, ask questions, and get responses in real time.

🔹 Tech Stack
Python – Backend language
Streamlit – Interactive UI
LangChain – LLM integration and RAG pipeline
FAISS – Efficient document retrieval
Google Gemini Embeddings – For text vectorization
Groq API (Gemma-2 9B IT) – AI model for answering questions
🔹 Workflow
Upload PDFs → The system loads and splits PDF text into chunks.
Create Embeddings → Text chunks are converted into vector embeddings using Google Gemini Embeddings.
Store in FAISS → The vectors are indexed for fast similarity search.
User Queries → User inputs a question in Streamlit.
Retrieve Relevant Context → FAISS searches for the most relevant document segments.
Generate Answer → The retrieved context is fed into Groq's Gemma-2 9B IT model to generate a response.
Display Response → The answer and relevant document excerpts are displayed.
🔹 Potential Use Cases
🔸 Legal Document Q&A – Search and summarize legal documents quickly.
🔸 Healthcare – Query medical reports and research papers.
🔸 Finance & Compliance – Retrieve financial statements and regulations efficiently.
🔸 Education & Research – Summarize academic papers and textbooks
