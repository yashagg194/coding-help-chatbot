from flask import Flask, request, jsonify, render_template
import openai
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings


app = Flask(__name__, template_folder="templates", static_folder="static")

# Set OpenAI API Key
OPENAI_API_KEY = "api key"

if not OPENAI_API_KEY:
    raise ValueError("Please set your OpenAI API key.")

# Load CSV Data
def load_documents():
    df1 = pd.read_csv("Software Questions.csv", encoding="Windows-1252")
    df2 = pd.read_csv("Geeks_For_Geeks_Questions_Dataset.csv", encoding="ascii")

    # Convert df1 rows to LangChain Document objects
    document_objects_csv1 = [
        Document(
            page_content=f"Question: {row['Question']}\nAnswer: {row['Answer']}",
            metadata={
                "source": "Software Questions.csv",
                "question_number": row.get('Question Number', 'N/A'),
                "category": row.get('Category', 'N/A'),
                "difficulty": row.get('Difficulty', 'N/A')
            }
        )
        for _, row in df1.iterrows()
    ]

    # Convert df2 rows to LangChain Document objects
    document_objects_csv2 = [
        Document(
            page_content=f"Question: {row['Question Name']}",
            metadata={
                "source": "Geeks_For_Geeks_Questions_Dataset.csv",
                "difficulty_level": row.get('Difficulty Level', 'N/A'),
                "submissions": row.get('Total Submissions', 'N/A'),
                "accuracy": row.get('Accuracy', 'N/A'),
                "company_tags": row.get('Company Tags', 'N/A')
            }
        )
        for _, row in df2.iterrows()
    ]

    return document_objects_csv1 + document_objects_csv2  # FIXED RETURN VALUE

# Initialize chatbot
def setup_chatbot():
    documents = load_documents()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    splits = text_splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(splits, OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))
    retriever = vectorstore.as_retriever()

    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)
    template = "Answer the question based on context:\n{context}\n\nQuestion: {question}"
    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

# Load chatbot on startup
chatbot = setup_chatbot()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "")

    if not user_input:
        return jsonify({"response": "Please enter a message."})

    bot_response = chatbot.invoke(user_input)
    return jsonify({"response": bot_response})

if __name__ == "__main__":
    app.run(debug=True)
