class BotMessageResponse:
    def __init__(self, responseText, state):
        self.responseText = responseText
        self.state = state
 
    def to_json(self):
        return {
            "responseText": self.responseText,
            "state": self.state
        }