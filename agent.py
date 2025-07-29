import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.document_loaders.pdf import PyPDFLoader
from langgraph.graph import StateGraph, END
from langchain.chains import RetrievalQA
from Util import create_filled_pdf, speech_to_text
from typing import List

# Load .env variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

# Step 1: Load Form Templates
form_templates = {
    "loan": {
        "path": "./docs/loan_form_template.pdf",
    },
    "credit_card": {
        "path": "./docs/credit_card_form_template.pdf",
    }
}

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
    "form_type": None
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
    print("🗣️ Which form do you want to fill? (loan / credit_card)")
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

def detect_next_field(state):
    remaining = [f for f in state["fields_required"] if f not in state["field_values"]]
    if remaining:
        return {**state, "current_field": remaining[0]}
    return state

def prompt_for_field(state):
    field = state["current_field"]
    user_input = speech_to_text(f"Please provide your {field}:")
    return {**state, "query": user_input}


def validate_and_store_field(state):

    field = state["current_field"]
    user_input = state["query"]

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
    prompt = f"""You are a strict data validator for a form field.

        Field: {field}
        Input: {user_input}

        Validation rules:
        - For 'dob':
            - Must be a valid date.
            - Acceptable formats: YYYY-MM-DD, DD-MM-YYYY, or 'DD Month YYYY'.
            - The date must be in the past (user must already be born).
        - For other fields:
            - Must be a valid input based on the field type.
            - For example, 'income' should be a positive number, 'credit_score' should be a number between 300 and 850, etc.

        If the input is valid, respond with the input exactly as provided.
        If the input is invalid, respond with only: invalid

        Examples:
        Input: 1992-04-28 (valid dob) → 1992-04-28
        Input: 28-04-1992 (valid dob) → 28-04-1992
        Input: 28 April 1992 (valid dob) → 28-04-1992
        Input: 2025-01-01 (future dob) → invalid
        Input: -5000 (income) → invalid
        Input: 750 (credit_score) → 750
    """

    response = model.generate_content(prompt).text.strip().lower()
    print("Model response:", response)  # Debug: See what the model returns

    if "invalid" in response:
        print(f"⚠️ Invalid input for {field}, please try again.")
        return state  # repeat same field

    updated_values = state["field_values"].copy()
    updated_values[field] = response
    return {**state, "field_values": updated_values}



def finalize_form(state):
    form_output = state["field_values"]
    form_type = state["form_type"]
    form_path = form_templates[form_type]["path"]
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

# Define the state graph
form_graph = StateGraph(FormState)
form_graph.add_node("select_form", select_form_type)
form_graph.add_node("detect_field", detect_next_field)
form_graph.add_node("ask_field", prompt_for_field)
form_graph.add_node("validate_store", validate_and_store_field)
form_graph.add_node("finalize", finalize_form)

form_graph.set_entry_point("select_form")
form_graph.add_edge("select_form", "detect_field")
form_graph.add_edge("detect_field", "ask_field")
form_graph.add_edge("ask_field", "validate_store")
form_graph.add_conditional_edges(
    "validate_store",
    lambda state: "detect_field" if len(state["field_values"]) < len(state["fields_required"]) else "finalize"
)
form_graph.add_edge("finalize", END)

compiled_graph = form_graph.compile()

# Step 5: Run
if __name__ == "__main__":
    final_state = compiled_graph.invoke(initial_state)
    print("\n📄 Final State:", final_state)
