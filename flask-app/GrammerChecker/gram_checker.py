from gramformer import Gramformer
import torch
from .error import map_error_types,get_error_types

def set_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


set_seed(1212)
gf = Gramformer(models = 1, use_gpu=False)
class GrammerResponse:
    def __init__(self, correctText, errorTypes):
        self.correctText = correctText
        self.errorTypes = errorTypes

def check_grammer(text):
    grammerResponse = GrammerResponse(None, None)
    influent_sentences=[text]
    for influent_sentence in influent_sentences:
        corrected_sentence = list(gf.correct(influent_sentence, max_candidates=1))[0]
        print("[Input] ", influent_sentence)
        print("[Corrected] ", corrected_sentence)
        if(influent_sentence != corrected_sentence):
            highlighted_text = gf.highlight(influent_sentence, corrected_sentence)
            print("[Highlighted] ", highlighted_text)
            reponse_text = map_error_types(highlighted_text)
            error_types= get_error_types(highlighted_text)
            print("[Error Types] ", error_types)
            print("[Response] ", reponse_text)
            grammerResponse.correctText = reponse_text
            grammerResponse.errorTypes = error_types
    return grammerResponse
