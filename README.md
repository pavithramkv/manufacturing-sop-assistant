# Manufacturing SOP Assistant Using Generative AI

## Project Overview

The Manufacturing SOP Assistant is a Generative AI-powered web application that helps users find information from manufacturing Standard Operating Procedure (SOP) documents. Users can upload a PDF and ask questions about its contents. The application retrieves relevant text and generates answers using Google Gemini.

## Features

- Upload manufacturing SOP documents in PDF format.
- Extract and split document text into smaller chunks.
- Generate embeddings using Google Gemini.
- Store and retrieve document information using ChromaDB.
- Ask questions about the uploaded SOP.
- Display AI-generated answers and relevant source text.
- Provide a simple web interface using Streamlit.

## Technologies Used

- Python
- Streamlit
- LangChain Community
- LangChain Text Splitters
- LangChain Google GenAI
- LangChain Chroma
- ChromaDB
- Google Gemini
- PyPDF

## Project Workflow

1. Upload an SOP PDF document.
2. Extract the text from the PDF.
3. Split the text into smaller chunks.
4. Generate embeddings and store the chunks in ChromaDB.
5. Retrieve relevant text based on the user's question.
6. Generate an answer using Google Gemini.
7. Display the answer and relevant source text.

## Installation and Setup

### Prerequisites

- Python installed on your computer.
- A Google Gemini API key.

### Steps

1. Clone or download this repository.
2. Open the project folder in a terminal.
3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project folder and add your API key:

   ```env
   GOOGLE_API_KEY=your_gemini_api_key
   ```

5. Start the application:

   ```bash
   streamlit run app.py
   ```

6. Open the local URL displayed in the terminal.

## Limitations

- Answers depend on the content available in the uploaded SOP.
- Scanned PDFs without readable text may not be processed correctly.
- AI-generated answers may occasionally contain unsupported information.
- The application requires a valid Google Gemini API key and internet access.

## Future Enhancements

- Add OCR support for scanned documents.
- Improve answer accuracy and source verification.
- Add secure user authentication.
- Support multiple SOP documents and persistent storage.
- Add more comprehensive testing and evaluation.

## Important Note

This application is intended to assist users in finding information from SOP documents. Always refer to the approved official SOP and follow workplace safety procedures.

## Author

Developed as a Generative AI project.
