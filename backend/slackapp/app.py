import os
import sys
import logging
from dotenv import load_dotenv
import re
import requests
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Setup path and environment variables
current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(kit_dir, ".."))
sys.path.extend([kit_dir, repo_dir])

env_path = os.path.join(repo_dir, ".env")
load_dotenv(env_path, override=True)

# Initialize Slack app
app = App(token=os.getenv("SLACK_TOKEN", ""))

# SambaQA API configuration
SAMBAQA_API_URL = os.getenv(
    "SAMBAQA_API_URL", "http://localhost:8000/retrievers/basic-rag/answer"
)
DEFAULT_COLLECTION = "snspubdocumentation"
DEFAULT_MODEL = "sambastudio/Meta-Llama-3-70B-Instruct"


def clean_input(text):
    # Remove Slack-style user mentions
    text = re.sub(r"^<@U[A-Z0-9]+>:?\s*", "", text.strip())

    # Remove non-printable characters
    text = "".join(char for char in text if char.isprintable())

    # Remove or replace problematic Unicode characters
    text = text.encode("ascii", "ignore").decode("ascii")

    # Remove extra whitespace
    text = " ".join(text.split())

    return text


def validate_input(text):
    if len(text) < 1 or len(text) > 1000:  # Adjust these limits as needed
        raise ValueError("Input must be between 1 and 1000 characters")

    # Add more validation rules as needed
    if not text.strip():
        raise ValueError("Input cannot be empty")

    return text


def query_sambaqa(question):
    logger.info(f"Original question: {question}")

    try:
        cleaned_question = clean_input(question)
        logger.info(f"Cleaned question: {cleaned_question}")

        validated_question = validate_input(cleaned_question)
        logger.info(f"Validated question: {validated_question}")

        payload = {
            "collection_name": DEFAULT_COLLECTION,
            "query": validated_question,
            "model_configuration": {
                "name": DEFAULT_MODEL,
                "parameters": {
                    "temperature": 0.1,
                    "do_sample": False,
                    "max_tokens_to_generate": 1024,
                    "select_expert": DEFAULT_MODEL.split("/")[-1],
                    "process_prompt": False,
                },
            },
            "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|> \nYou are an AI assistant specializing in accurate information retrieval and analysis. Your primary goal is to provide reliable answers based solely on the given context. Follow these guidelines:\n\n1. Answer Step-by-Step:\n   - Break down your reasoning process.\n   - Explain how each piece of information relates to the question.\n\n2. Source Attribution:\n   - ALWAYS cite sources for any information you use in your response.\n   - Use the following format for citations:\n     a) For direct quotes: [Quote: \"exact text\"]\n     b) For paraphrased information: [Source: summary of information]\n     c) For weblinks: [Link: full URL]\n   - When possible, include any additional source details provided, such as author, date, or title.\n\n3. Accuracy Check:\n   - After formulating your answer, review it against the context.\n   - Ensure every statement is supported by the provided information.\n\n4. Handling Uncertainty:\n   - If the context doesn't contain enough information, say \"The given context does not provide sufficient information to answer this question fully.\"\n   - For partial answers, clearly state what you can and cannot answer based on the context.\n\n5. No External Knowledge:\n   - Do not use any information beyond the given context.\n   - If tempted to add external information, stop and reassess.\n\n6. Avoid Assumptions:\n   - Do not infer or extrapolate beyond what's explicitly stated in the context.\n   - If the question requires assumptions, clearly state them as such.\n\nRemember: It's better to provide a partial answer or admit lack of information than to give an inaccurate or unsupported response.\n\nContext: {context}\n\nQuestion: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
            "retriever_config": {
                "compressor_model_provider": "mixedbread-ai",
                "compressor_model_name": "mixedbread-ai/mxbai-rerank-xsmall-v1",
                "top_k": 12,
                "search_type": "similarity",
                "search_kwargs": {"k": 20},
                "filter": {},
            },
            "stream": False,
        }

        response = requests.post(SAMBAQA_API_URL, json=payload)
        response.raise_for_status()
        return response.json()["answer"]
    except ValueError as e:
        logger.error(f"Input validation error: {str(e)}")
        return f"Error: {str(e)}"
    except requests.RequestException as e:
        logger.error(f"Error querying SambaQA: {str(e)}")
        return f"Error querying SambaQA: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error in query_sambaqa: {str(e)}", exc_info=True)
        return "An unexpected error occurred. Please try again later."


@app.event("message")
def handle_message(message, say):
    try:
        if message["channel_type"] == "im":
            question = message["text"]
            response = query_sambaqa(question)
            say(response)
    except Exception as e:
        error_message = (
            "SambaQA is experiencing difficulties, hang tight while we investigate."
        )
        logger.error(f"Error in handle_message: {str(e)}", exc_info=True)
        say(error_message)


@app.event("app_mention")
def handle_app_mention_events(body, say):
    try:
        question = body["event"]["text"]
        response = query_sambaqa(question)
        say(response)
        logger.info(f"App mention event body: {body}")
    except Exception as e:
        error_message = (
            "SambaQA is experiencing difficulties, hang tight while we investigate."
        )
        logger.error(f"Error in handle_app_mention_events: {str(e)}", exc_info=True)
        say(error_message)


if __name__ == "__main__":
    try:
        logger.info("Starting Slack app...")
        SocketModeHandler(app, os.getenv("SLACK_APP_TOKEN", "")).start()
    except Exception as e:
        logger.error(f"Error starting Slack app: {str(e)}", exc_info=True)
