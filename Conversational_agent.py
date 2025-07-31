import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.chat_models import init_chat_model
from Util import speech_to_text, speak_text

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set up a conversation agent powered by Gemini, Claude, or another chat LLM
def init_conversation_agent():
    # Use Gemini or other LLM of choice; can add tools here too!
    return init_chat_model("gpt-4o-mini", model_provider="openai", openai_api_key=openai_api_key)

conversation_agent = init_conversation_agent()

def conversational_speech_agent(system_prompt):
    """Intelligent agent-driven speech prompt/response."""
    # The agent can give spoken instructions, ask to rephrase, etc.
    speak_text(system_prompt)

    user_input = speech_to_text()  # Gets one round of spoken input

    # Optionally, pass both system prompt and user input to LLM for clarification
    response = conversation_agent([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input)
    ])
    speak_text(response.content)
    # You may choose to just return user_input or use the agent's LLM reply
    return response.content.strip()
