# Chat with PDF Files

This project allows users to upload PDF files and ask questions about their contents using a chat interface. The application processes the PDF files, extracts their text, and provides answers to user questions based on the content of the PDFs.

## Features

- Upload multiple PDF files.
- Extract text from PDF files.
- Ask questions about the PDF content via a chat interface.
- Display the PDF on the left side and the chat interface on the right side.

## Technologies Used

- Python
- Flask
- PyPDF2
- LangChain
- Google Gemini Pro
- FAISS
- HTML/CSS
- JavaScript

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/habeebmoosa/PdfQnA.git
    cd chat-with-pdf
    ```

2. Create and activate a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

4. Create a `.env` file in the project root directory and add your Google API key:

    ```
    GOOGLE_API_KEY=your_google_api_key_here
    ```

## Usage

1. Start the Flask application:

    ```bash
    python app.py
    ```

2. Open your web browser and navigate to `http://127.0.0.1:5000/`.

3. Upload your PDF file using the provided form.

4. Ask questions about the content of the uploaded PDFs via the chat interface.

## Contributing

Contributions are welcome! Please create a pull request or open an issue to discuss your ideas.

## License

This project is licensed under the MIT License.

