# Summarizer_and_Chatbot

Welcome to the LangChain Summarizer GitHub repository! This project utilizes OpenAI's language model to provide file summarization and question-answering capabilities. Users can input a file and specify the desired word limit for the summary. Additionally, they can ask questions about the contents of the file using the OpenAI API.

## Setup

To get started with LangChain Summarizer, you need to complete the following setup steps:

### 1. Milvus Connection

1. Install Milvus by following the instructions provided in the [Milvus documentation](https://milvus.io/docs/v2.0.0/install_standalone-docker.md).
2. Start Milvus server by running the appropriate command for your setup.
3. Connect to Milvus in your code by providing the necessary connection parameters (host, port, etc.). You can refer to the Milvus documentation for code examples and API usage.

### 2. OpenAI API

1. Sign up for an account on the [OpenAI website](https://openai.com/).
2. Retrieve your OpenAI API key from the developer dashboard.
3. Store the API key securely as an environment variable or in a configuration file.

## Usage

1. Clone this repository to your local machine using the following command:
   ```
   git clone https://github.com/yashika2406/Summarizer_and_Chatbot.git
   ```

2. Install the required dependencies by navigating to the project directory and running:
   ```
   pip install -r requirements.txt
   ```

3. Start the LangChain Summarizer application by running the main script:
   ```
   python notebooks/chatbot.py
   ```
4. First, it will ask whether you want summarization or not.
   
5. If yes, Enter the file path and specify the desired word limit for the summary.

6. After that, chatbot will be started. First enter the query, then it will use the OpenAI API to generate an answer based on the uploaded file.


## Acknowledgments

We would like to express our gratitude to the open-source community for their contributions, and to OpenAI for their powerful language model, which forms the core of this project.

Please feel free to reach out with any questions or feedback. Happy summarizing!
