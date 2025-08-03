import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.document_loaders.pdf import PyPDFLoader
from langgraph.graph import StateGraph, END
from typing import Dict, List
from typing import List
from validationModel import init_validation_agent
from Conversational_agent import conversational_speech_agent
from Util import create_filled_pdf, get_form_templates


# Load .env variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

# Step 1: Load Form Templates
form_templates = get_form_templates(os.getenv("FORM_TEMPLATES"))

# Step 2: Define State
initial_state = {
    "query": "",
    "docs": [],
    "answer": "",
    "confidence": 0.0,
    "form": {},
    "fields_required": [],
    "field_values": {},
    "current_field": None,
    "form_type": None,
    "current_field_valid": None,
}

def extract_fields_from_pdf(form_path: str) -> List[str]:
    # Load PDF pages
    loader = PyPDFLoader(form_path)
    documents = loader.load_and_split()

    # Merge and truncate PDF text content
    full_content = "\n\n".join([doc.page_content for doc in documents])[:8000]

    # Create manual prompt
    manual_prompt = (
        "You're analyzing a form PDF. Based on the following content, extract and return only the list of form fields "
        "that a user is expected to fill out.\n\n"
        "Respond ONLY with a valid Python list of strings. Do NOT include any explanation, markdown, or code formatting.\n\n"
        f"{full_content}"
    )

    try:
        # Initialize Gemini model
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
        response = model.generate_content(manual_prompt).text.strip()

        print("🧠 Gemini Response:", response)

        # Try to parse the response into a Python list
        fields = eval(response)
        if isinstance(fields, list):
            return [f.lower() for f in fields]
    except Exception as e:
        print("❌ Failed to extract fields:", e)

    return []


# Step 3: Define Nodes

def select_form_type(state):
    print("🗣️ Which form do you want to fill? ")
    form_type = input("You: ").strip().lower()
    if form_type not in form_templates:
        print("⚠️ Invalid form type, defaulting to 'loan'")
        form_type = "loan"

    selected_form = form_templates[form_type]
    form_path = selected_form["path"]

    fields = extract_fields_from_pdf(form_path)
    print("📝 Extracted fields:", fields)

    return {
        **state,
        "form_type": form_type,
        "form": {"name": f"{form_type.title()} Form"},
        "fields_required": fields,
    }

def conversational_orchestration(state) -> Dict:
    """
    Orchestrator node that selects the next field to fill,
    invokes the conversational agent loop to get and confirm value,
    updates state and conversation history accordingly.
    """

    required_fields: List[str] = state.get("fields_required", [])
    field_values: Dict[str, str] = state.get("field_values", {})
    # Determine remaining fields to fill
    remaining_fields = [f for f in required_fields if f not in field_values]

    # If no remaining fields, return state to proceed to finalize
    if not remaining_fields:
        return state

    current_field = remaining_fields[0]
    validation_status = state.get("current_field_valid")

    # Build minimal context for conversational agent
    context = {
        "field": current_field,
        "fields_filled": field_values,
        "validation_failed": validation_status,
    }

    # Run the interactive agent dialog loop to get confirmed input and updated history
    confirmed_value = conversational_speech_agent(context)

    # Update state with returned values
    # Reset validation flag for next iteration
    return {
        **state,
        "current_field": current_field,
        "query": confirmed_value,
        "current_field_valid": validation_status,
    }


def validate_and_store_field(state):
    field = state["current_field"]
    user_input = state["query"]

    # Build a user message the agent can interpret
    message = (
        f"Field: {field}\n"
        f"Value: {user_input}\n"
        "Please validate this value appropriately using your tools."
    )

    # Prepare agent input
    agent_input = {
        "messages": [{"role": "user", "content": message}]
    }

    # Invoke the ReAct validation agent (this calls the appropriate tool internally)
    validation_agent = init_validation_agent()
    result = validation_agent.invoke(agent_input)
    response = result["messages"][-1].content.strip().lower()

    print(f"Agent validation response for '{field}': {response}")

    if response == "invalid":
        print(f"⚠️ Invalid input for field '{field}', please try again.")
        return {**state, "current_field_valid": False}  # Keep same field, await corrected input

    # Save validated result
    updated_values = state["field_values"].copy()
    updated_values[field] = response
    return {**state, "field_values": updated_values, "current_field_valid": True}


def finalize_form(state):
    form_output = state["field_values"]
    form_type = state["form_type"]
    filled_path = f"./filled_forms/{form_type}_filled"

    create_filled_pdf(filled_path, form_output)

    print("\n✅ Form filled with:")
    for k, v in form_output.items():
        print(f"{k}: {v}")
    print(f"\n📄 Saved filled form at: {filled_path}")

    return {**state, "answer": f"Form saved to {filled_path}"}


# Step 4: Build Graph
from typing import TypedDict, List, Dict, Optional, Union

class FormState(TypedDict):
    query: str
    docs: List
    answer: str
    confidence: float
    form: Dict
    fields_required: List[str]
    field_values: Dict[str, str]
    current_field: Optional[str]
    form_type: Optional[str]
    current_field_valid: Optional[bool]

# Define the state graph
form_graph = StateGraph(FormState)
form_graph.add_node("select_form", select_form_type)
form_graph.add_node("validate_store", validate_and_store_field)
form_graph.add_node("finalize", finalize_form)
form_graph.add_node("conversational_orchestration", conversational_orchestration)

form_graph.set_entry_point("select_form")
form_graph.add_edge("select_form", "conversational_orchestration")
form_graph.add_edge("conversational_orchestration", "validate_store")
form_graph.add_conditional_edges(
    "validate_store",
    lambda state: "conversational_orchestration" if len(state["field_values"]) < len(state["fields_required"]) else "finalize"
)
form_graph.add_edge("finalize", END)

compiled_graph = form_graph.compile()

# Step 5: Run
if __name__ == "__main__":
    final_state = compiled_graph.invoke(initial_state, config={"verbose": True, "recursion_limit": 100})
    print("\n📄 Final State:", final_state)
