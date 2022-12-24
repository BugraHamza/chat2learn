import re

error_dict: dict = {
    "VERB:SVA": "Deneme 1",
    "VERB": "Deneme 2",
}

def map_error(error_type):
    regex_to_find_type_value = re.compile(r"(?<=type=')[^']+(?=')")
    error_type_value = re.findall(regex_to_find_type_value, error_type)[0]
    try:
        return f"{error_type} description='{error_dict[error_type_value]}'"
    except:
        return f"{error_type} description='Unknown error type'"
    return new_tags

def map_error_types(highlighted_text):
    regex_to_find_type = re.compile(r"(type='.*?')")
    error_types = re.findall(regex_to_find_type, highlighted_text)
    for error_type in error_types:
        highlighted_text = highlighted_text.replace(error_type,map_error(error_type))
    return highlighted_text


def get_error_types(highlighted_text):
    regex_to_find_type = re.compile(r"type='(.*?)'")
    error_types = re.findall(regex_to_find_type, highlighted_text)
    return error_types


