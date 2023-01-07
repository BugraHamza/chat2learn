import re

error_dict: dict = {
    "VERB:SVA": "Ozneyle eylem arasinda uyumsuzluk var",
    "ORTH": "Yazim yanlişi",
    "PUNCT": "Noktalama işareti hatası",
    "DET": "Belirleyici hatası. ",
    "ADJ": "Sıfat",
    "PRON": "Zamir hatası",
    "ADV": "Zarf",
    "VERB": "Fiil hatası",
    "NUM": "Sayı",
    "CONJ": "Bağlaç",
    "WO": "Kelime sırası yanlışı",
    "NOUN:POSS": "Görünüşe göre bu isim formu yanlış olabilir.",
    "SPELL":"Yazım Yanlışı",
    "OTHER":"Diğer",
    "VERB:FORM":"Fiil biçimi yanlışı",
    "NOUN:NUM":"İsim sayı yanlışı",
    "PREP":"Eksik ya da fazla prepozisyon kullanımı",
    
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
            error_type_list.append({"code":error_type,"description":"Belirsiz hata tipi"})
    return error_type_list


