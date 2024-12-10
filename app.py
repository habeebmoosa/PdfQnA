import os
from flask import Flask, request, render_template, redirect, url_for, jsonify
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def get_pdf_text(pdf_files):
    """Extract text from uploaded PDF files."""
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Split text into manageable chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

def create_vector_store(chunks):
    """Create a temporary vector store from text chunks."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.from_texts(chunks, embedding=embeddings)

def get_conversational_chain():
    """Create the conversational QA chain."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. 
    If the answer is not in the provided context, just say "answer is not available in the context", 
    don't provide a wrong answer.

    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", client=genai, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)

def process_user_question(vector_store, user_question):
    """Process user question against the vector store."""
    # Find similar documents
    docs = vector_store.similarity_search(user_question)

    # Create chain and get response
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response

@app.route("/", methods=["GET", "POST"])
def index():
    """Main index route to handle PDF uploads."""
    if request.method == "POST":
        # Clear previous uploads
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        # Save new PDFs
        pdf_files = request.files.getlist("pdf_files")
        for pdf in pdf_files:
            pdf.save(os.path.join(app.config['UPLOAD_FOLDER'], pdf.filename))
        
        return redirect(url_for("chat"))

    return render_template("index.html")

@app.route("/chat", methods=["GET", "POST"])
def chat():
    """Chat route to handle questions and responses."""
    if request.method == "POST":
        # Get uploaded PDF files
        pdf_files = [os.path.join(app.config['UPLOAD_FOLDER'], f) for f in os.listdir(app.config['UPLOAD_FOLDER'])]
        
        if not pdf_files:
            return jsonify({"answer": "No PDF files uploaded. Please upload a PDF first."})

        # Extract text from PDFs
        raw_text = get_pdf_text([open(f, 'rb') for f in pdf_files])
        
        # Create text chunks and vector store
        text_chunks = get_text_chunks(raw_text)
        vector_store = create_vector_store(text_chunks)
        
        # Process user question
        user_question = request.json.get("question")
        response = process_user_question(vector_store, user_question)
        
        return jsonify({"answer": response['output_text']})
    
    return render_template("chat.html")

@app.route("/pdf", methods=["GET"])
def pdf():
    """Get uploaded PDF filenames."""
    pdf_files = os.listdir(app.config['UPLOAD_FOLDER'])
    if pdf_files:
        return jsonify({"pdf_url": pdf_files[0] if pdf_files else ""})
    return jsonify({"pdf_url": ""})

if __name__ == "__main__":
    app.run(debug=True)