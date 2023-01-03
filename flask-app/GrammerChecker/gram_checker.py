from .gramformer import Gramformer
import torch
from .error import map_error_types,get_error_types
import evaluate
rouge = evaluate.load('rouge')
def set_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


set_seed(1212)
gf = Gramformer(models = 1, use_gpu=False)
class GrammerResponse:
    def __init__(self,correctedSentence, taggedText, errorTypes,score):
        self.taggedText = taggedText
        self.errorTypes = errorTypes
        self.correctedSentence = correctedSentence
        self.score = score


def check_grammer(text):
    grammerResponse = GrammerResponse(None, None,None,None)
    influent_sentences=[text]
    for influent_sentence in influent_sentences:
        corrected_sentence_set = gf.correct(influent_sentence, max_candidates=42)
        print("[Corrected Sentence Set] ", corrected_sentence_set)
        corrected_sentence = list(corrected_sentence_set)[0]
        references = [list(corrected_sentence_set)]
        predictions = [text]
        results = rouge.compute(predictions=predictions,references=references)
        print("[Score] ", results)
        grammerResponse.score = round(results['rougeL'],2)
        print("[Input] ", influent_sentence)
        print("[Corrected] ", corrected_sentence)
        if(influent_sentence != corrected_sentence):
            highlighted_text = gf.highlight(influent_sentence, corrected_sentence)
            print("[Highlighted] ", highlighted_text)
            reponse_text = map_error_types(highlighted_text)
            error_types= get_error_types(highlighted_text)
            print("[Error Types] ", error_types)
            print("[Response] ", reponse_text)
            grammerResponse.correctedSentence = corrected_sentence
            grammerResponse.taggedText = reponse_text
            grammerResponse.errorTypes = error_types
    return grammerResponse
