# Chat with Your Data using Ollama

This project demonstrates how to create a question-answering system that can chat with your PDF data using Ollama, LangChain, and various other libraries.

## Features

- Load and process PDF documents (local or online)
- Split documents into manageable chunks
- Create embeddings and store them in a vector database
- Perform similarity search to retrieve relevant context
- Generate responses using Ollama's language model

## Requirements

- Python 3.7+
- LangChain
- Chroma
- GPT4All
- Ollama
- PyPDF2

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install langchain chromadb gpt4all ollama pypdf2
   ```
3. Make sure you have Ollama installed and running on your system

## Usage

1. Place your PDF file in the `data/` directory or use an online PDF URL
2. Update the `loader` variable in `main.py` to point to your PDF file
3. Run the script:
   ```
   python main.py
   ```
4. Enter your queries when prompted
5. Type "exit" to quit the program

## How it Works

1. The script loads a PDF document and splits it into smaller chunks
2. It creates embeddings for these chunks and stores them in a Chroma vector database
3. When you enter a query, it retrieves the most relevant chunks from the database
4. The retrieved context is then used to generate a response using Ollama's language model
5. The response is streamed to the console

## Customization

- You can adjust the `chunk_size` and `chunk_overlap` parameters in the `RecursiveCharacterTextSplitter` to change how the document is split
- Modify the prompt template to change how the model generates responses
- Change the Ollama model by updating the `model` parameter in the `Ollama` constructor

## License

[MIT License](LICENSE)

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check [issues page](https://github.com/yourusername/chat-with-your-data-ollama/issues) if you want to contribute.

## Show your support

Give a ⭐️ if this project helped you!
