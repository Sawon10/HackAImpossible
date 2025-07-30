import fitz  # PyMuPDF
import os
import json
import speech_recognition as sr
from datetime import date

def get_form_templates(form_templates_raw):
    try:
        form_paths = json.loads(form_templates_raw)
        form_templates = {
            form_type: {"path": path}
            for form_type, path in form_paths.items()
        }
    except json.JSONDecodeError as e:
        print("❌ Failed to parse FORM_TEMPLATES from .env:", e)
    return form_templates

def create_filled_pdf(output_path, field_values):
    # Create a new PDF document
    doc = fitz.open()
    page = doc.new_page()
    name = ""
    today = date.today().strftime("%Y-%m-%d")

    # Write each field and value
    y = 100
    for field, value in field_values.items():
        if field == "name":
            name = value
        page.insert_text((100, y), f"{field.title()}: {value}", fontsize=12)
        y += 25

    # Save to output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc.save(output_path  + name + today + ".pdf")
    doc.close()



def speech_to_text(prompt="Speak now..."):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print(f"🗣️ {prompt}")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        print("❌ Could not understand audio.")
        return ""
    except sr.RequestError:
        print("❌ Could not request results from Google Speech Recognition service.")
        return ""

