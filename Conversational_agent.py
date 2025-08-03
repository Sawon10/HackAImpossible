import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from typing import List, Dict, Tuple
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

# Assume the following imports exist in your module:
# from Util import speech_to_text, speak_text
# from your existing LLM init: conversation_agent

def conversational_speech_agent(context: Dict[str, object]) -> str:
    """
    Converses interactively with the user to get and confirm a value for a single form field.

    Args:
        context: 
          - 'field' (str): current field name to ask for
          - 'fields_filled' (dict): previously filled fields
          - 'validation_failed' (bool): if true, last entry was invalid
        conversation_history: List of LangChain messages (SystemMessage/HumanMessage) for context

    Returns:
        confirmed_value (str): the final confirmed value for the field
        updated_conversation_history (List[BaseMessage]): conversation history updated with new turns
    """

    field = context["field"]
    filled_fields = context.get("fields_filled", {})
    validation_failed = context.get("validation_failed", False)

    extracted_value = None
    if validation_failed:
        prompt_for_field = (
            f"you were filing out the {field} field, but the last value you provided was invalid. "
            f"The previous value user provided for '{field}' was invalid or unclear. "
            f"Please ask the user nicely for his {field} again."
     )
    elif len(filled_fields) == 0:
        prompt_for_field = (f"you are filling out the {field} for a form, "
                           f"Please ask the user for their {field}."
                           f"remember this question is going to be start of the conversation"
                           f"do not include sure or okay in the question.")
    else:
        prompt_for_field = (f"you were filing out the {field} for a form, "
                           f"Please ask the user for their {field}.")
    while True:
        question_for_field = conversation_agent([SystemMessage(content=prompt_for_field)])
        speak_text(question_for_field.content)  # Speak the prompt

        # Listen to user
        user_input = speech_to_text().strip()
        # Use LLM to extract only the value part for the field
        extract_prompt = (
            f"The user said: '{user_input}'. "
            f"Extract and return only the {field} as a concise value, nothing else."
        )
        extract_response = conversation_agent([
            SystemMessage(content=extract_prompt),
            HumanMessage(content=user_input)
        ])
        extracted_value = extract_response.content.strip()

        # Confirm extracted value with the user
        confirm_prompt = (f"The extracted field for the {field} is '{extracted_value}'. " 
                          f"please genarate a confirmation question for the user to confirm this value.")
        confirmation_question_for_user = conversation_agent([SystemMessage(content=confirm_prompt)])
        speak_text(confirmation_question_for_user.content)  # Speak the confirmation question
        # Get user confirmation
        user_confirm = speech_to_text().strip().lower()

        check_user_confirmation_prompt = (
            f"providing the confirmation question we asked in the last step that is {confirmation_question_for_user.content},"
            f"please check if the user confirmed the value or not based on the question"
            f"If user confirmed the value return only 'yes' if user did not confirm return 'no' as a response."
        )
        # Check if user confirmation contains any of the confirmation keywords
        validate_user_confirmation = conversation_agent([SystemMessage(content=check_user_confirmation_prompt),
            HumanMessage(content=user_confirm)])
        if validate_user_confirmation.content.strip().lower() == "yes":
            # Confirmed; return value and updated history
            return extracted_value
        else:
            # Not confirmed, apologize and repeat loop
            apology_propt = "Sorry for the misunderstanding. Let's try again."
            speak_text(apology_propt)
            prompt_for_field = (
                f"you were filing out the {field} field, but the last value was not confirmed. "
                f"Please ask the user nicely for his {field} again."
            )
    