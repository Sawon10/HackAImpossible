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

def init_conversation_agent():
    # Initialize your LLM chat model instance
    return init_chat_model("gpt-4o-mini", model_provider="openai", openai_api_key=openai_api_key)

conversation_agent = init_conversation_agent()


def conversational_speech_agent(system_prompt=None, state=None, messages=None):
    """
    Context-aware conversational speech agent.

    Args:
        system_prompt (str): Optional prompt. If not set and state is given, builds a rich prompt from state.
        state (dict): Current form state with form_type, fields_required, field_values.
        messages (list): List of SystemMessage and HumanMessage for LLM context.

    Returns:
        str: The LLM's spoken response content.
    """
    # If messages not provided, create contextually from state/system_prompt
    if messages is None:
        if system_prompt is None and state is not None:
            form_name = state.get("form_type", "the form").title() if state else "the form"
            filled = state.get("field_values", {}) if state else {}
            remaining = [f for f in state.get("fields_required", []) if f not in filled] if state else []
            filled_summary = ", ".join(f"{k}: {v}" for k, v in filled.items()) if filled else "none"
            remaining_summary = ", ".join(remaining) if remaining else "none"
            system_prompt = (
                f"You are assisting to fill out the form '{form_name}'. "
                f"Fields filled so far: {filled_summary}. "
                f"Remaining fields: {remaining_summary}. "
                f"Please ask the user for the next missing field clearly."
            )
        messages = [SystemMessage(content=system_prompt)]

    # Speak only the most recent system prompt (not all history)
    speak_text(messages[-1].content)

    # Get user spoken input
    user_input = speech_to_text()
    messages.append(HumanMessage(content=user_input))

    # Send messages to LLM agent for clarification/elaboration
    response = conversation_agent(messages)

    # Speak out the agent's spoken reply
    speak_text(response.content)
    return user_input