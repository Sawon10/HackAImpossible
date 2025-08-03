from langchain_core.tools import tool
from datetime import datetime
import re
import dateparser

@tool
def validate_name(value: str) -> str:
    """
    Validates a typical full name.
    Accepts alphabetic characters, spaces, hyphens, apostrophes.
    Returns the cleaned name or "invalid".
    """
    if not value or not isinstance(value, str):
        return "invalid"
    
    # Basic cleanup: strip spaces
    value = value.strip()
    
    # Reject if contains numbers or strange characters
    # Allow letters, spaces, hyphens (-), apostrophes (')
    if re.fullmatch(r"[A-Za-z\s\-\']+", value):
        # Normalize: title case
        cleaned = " ".join(word.capitalize() for word in value.split())
        return cleaned
    
    return "invalid"


@tool
def validate_date(value: str) -> str:
    """
        Validates a date string that may come from speech input in various spoken formats.
        Returns normalized date string 'YYYY-MM-DD' if valid and not in the future, else 'invalid'.
    """
    if not value or not isinstance(value, str):
        return "invalid"

    # Normalize common speech input quirks:
    # 1. Convert words like "dash", "slash", "dot" to separators
    normalized = value.lower()
    normalized = re.sub(r"\b(dash|slash|dot|hyphen)\b", "-", normalized)
    normalized = re.sub(r"[.,]", "-", normalized)  # replace dots/commas with dash
    # 2. Remove ordinal suffixes (1st, 2nd, 3rd, 4th, ...)
    normalized = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", normalized)
    # 3. Remove extra spaces
    normalized = re.sub(r"\s+", " ", normalized).strip()

    # Use dateparser local to parse loosely formatted date strings
    parsed_dt = dateparser.parse(normalized, settings={'PREFER_DATES_FROM': 'past'})

    if not parsed_dt:
        return "invalid"

    # Check if date is in the future and reject if so
    if parsed_dt.date() > datetime.now().date():
        return "invalid"

    # Return normalized ISO date string
    return parsed_dt.strftime("%Y-%m-%d")


@tool
def validate_id(value: str) -> str:
    """Validates alphanumeric IDs of length 5–15. Returns formatted ID or 'invalid'."""
    stripped = value.strip().upper()
    if re.fullmatch(r"[A-Z0-9]{5,15}", stripped):
        return stripped
    return "invalid"

@tool
def validate_numeric(value: str, min_value: float = 0, max_value: float = 1e6) -> str:
    """Validates numeric value in [min_value, max_value]. Returns value or 'invalid'."""
    try:
        num = float(value)
        if min_value <= num <= max_value:
            return str(int(num) if num.is_integer() else num)
    except Exception:
        pass
    return "invalid"

@tool
def validate_address(value: str) -> str:
    """
    Validates an address (simple: must be at least 10 characters, contain letters and numbers).
    Returns the cleaned address or 'invalid'.
    """
    if not value or not isinstance(value, str):
        return "invalid"
    value = value.strip()
    # Simple requirement: address has at least 10 chars, some letters, some digits
    if len(value) >= 10 and re.search(r"[A-Za-z]", value) and re.search(r"\d", value):
        return value
    return "invalid"

