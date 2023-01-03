import re

error_dict: dict = {
    "VERB:SVA": "Özneyle eylem arasında uyumsuzluk",
    "ORTH": "Yazım yanlışı",
    "PUNCT": "Noktalama işareti",
    "DET": "Sıfat",
    "ADJ": "Sıfat",
    "NOUN": "İsim",
    "PRON": "Zarf",
    "ADV": "Sıfat",
    "NUM": "Sayı",
    "CONJ": "Bağlaç",
    "AUX": "Yardımcı fiil",
    "INTJ": "Ünlem",
    "PART": "Zarf",
    "WO": "Kelime sırası yanlışı",
    "NOUN:POSS": "Görünüşe göre bu isim formu yanlış olabilir."
    
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
    error_type_list = []
    regex_to_find_type = re.compile(r"type='(.*?)'")
    error_types = re.findall(regex_to_find_type, highlighted_text)
    for error_type in error_types:
        try:
            error_type_list.append({"code":error_type,"description":error_dict[error_type]})
        except:
            error_type_list.append({"code":error_type,"description":"Unknown error type"})
    return error_type_list


