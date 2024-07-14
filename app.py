import os
from flask import Flask, request, render_template, redirect, url_for, jsonify, send_from_directory
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

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks

def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", client=genai, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        pdf_files = request.files.getlist("pdf_files")
        for pdf in pdf_files:
            pdf.save(os.path.join(app.config['UPLOAD_FOLDER'], pdf.filename))
        raw_text = get_pdf_text(pdf_files)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)
        return redirect(url_for("chat"))

    return render_template("index.html")

@app.route("/chat", methods=["GET", "POST"])
def chat():
    if request.method == "POST":
        user_question = request.json.get("question")
        response = user_input(user_question)
        return jsonify({"answer": response['output_text']})
    
    return render_template("chat.html")

@app.route("/pdf", methods=["GET"])
def pdf():
    pdf_files = os.listdir(app.config['UPLOAD_FOLDER'])
    if pdf_files:
        return jsonify({"pdf_url": url_for('uploaded_file', filename=pdf_files[0])})
    return jsonify({"pdf_url": ""})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)