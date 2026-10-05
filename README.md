Local PDF RAG Chatbot
This project is a local Retrieval-Augmented Generation (RAG) application that allows users to upload a PDF and ask questions about its content.
The system uses Python, Streamlit, LangChain, Ollama, Llama 3.1, ChromaDB, and DeepEval. Everything runs locally, so no paid API is required.
Features:
- Upload a PDF file
- Extract text from the PDF
- Split the document into smaller chunks
- Create embeddings locally
- Store document chunks in ChromaDB
- Retrieve the most relevant chunks for a question
- Generate answers with Llama 3.1
- Show retrieved source content
- Choose the number of retrieved chunks
- Choose between concise, detailed, and step-by-step answer styles
- Evaluate the RAG system with DeepEval
How it works:
PDF → Text extraction → Text chunks → Embeddings → ChromaDB → User question → Relevant chunks retrieved → Llama 3.1 → Answer
When a user asks a question, ChromaDB searches for the most relevant parts of the uploaded PDF. These chunks are then passed to Llama 3.1 together with the question. The model is instructed to answer only using information from the document.
Project files:
app.py
evaluate.py
requirements.txt
README.md
.gitignore
Installation:
1. Install Ollama from:
   https://ollama.com/
2. Download the Llama model:
   ollama pull llama3.1
3. Download the embedding model:
   ollama pull nomic-embed-text
4. Install Python dependencies:
   pip install -r requirements.txt
5. Run the Streamlit application:
   streamlit run app.py
After that, open the local Streamlit page in your browser, upload a PDF, click “Process PDF”, and start asking questions.
Example question:
Who is the author?
The system searches the uploaded document, retrieves the relevant context, and generates an answer based on the PDF.
RAG Evaluation
The project also includes evaluation using DeepEval.
The following metrics were used:
- Answer Relevancy
- Faithfulness
- Contextual Relevancy
Evaluation results:
Answer Relevancy: 0.67
Faithfulness: 1.00
Contextual Relevancy: 0.89
Overall pass rate: 100%
3 out of 3 test cases passed.
To configure DeepEval with Ollama:
deepeval set-ollama --model=llama3.1
To run the evaluation:
python evaluate.py
Why RAG?
A normal language model answers using its existing knowledge. A RAG system first retrieves relevant information from an external source, such as a PDF, and then uses this information to generate the answer. This makes the answer more connected to the actual document and helps reduce hallucinations.
The project runs locally with Ollama, so the PDF content and user questions do not need to be sent to a paid external LLM API.
Author:
Maksim Brusilovskii
