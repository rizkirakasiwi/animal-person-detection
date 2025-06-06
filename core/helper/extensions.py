def to_camel_case(text: str) -> str:
    parts = text.split()
    return parts[0].lower() + ''.join(word.capitalize() for word in parts[1:])