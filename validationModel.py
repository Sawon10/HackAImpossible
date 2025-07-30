from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from tools import validate_date, validate_id, validate_numeric, validate_name
from dotenv import load_dotenv
import os

openai_api_key = os.getenv("OPENAI_API_KEY")

def init_validation_agent():
    # Initialize your chat model
    model = init_chat_model("gpt-4o-mini", model_provider="openai", openai_api_key = openai_api_key)

    # Define system prompt controlling tool selection logic
    validation_prompt = """
        You are a smart validation assistant for form fields. For each input, decide which validation tool to use:
        - For date fields (dob, birthdate), use 'validate_date'.
        - For numeric fields (income, salary, credit_score), use 'validate_numeric' with appropriate min/max values.
        - For ID fields (ssn, passport_id, pan), use 'validate_id'.
        Only respond with the validated value or 'invalid', nothing else.
    """

    # Register the validator tools
    tools = [validate_date, validate_id, validate_numeric, validate_name]

    # Create the ReAct agent
    return create_react_agent(model, tools=tools, prompt=validation_prompt)