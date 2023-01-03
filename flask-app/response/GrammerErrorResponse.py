class GrammerErrorResponse:
    def __init__(self,taggedCorrectText, correctText, error_types,score):
        self.taggedCorrectText = taggedCorrectText
        self.correctText = correctText
        self.errorTypes = error_types
        self.score = score
 
    def to_json(self):
        return {
            "taggedCorrectText": self.taggedCorrectText,
            "correctText": self.correctText,
            "errorTypes": self.errorTypes,
            "score": self.score
        }