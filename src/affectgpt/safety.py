import re
CRISIS = re.compile(r'\b(self\s*harm|suicide|kill myself|end it all|overdose)\b', re.I)
def check(text: str):
    if CRISIS.search(text):
        return ('I am really sorry you are going through this. You are not alone. '
                'If you are in immediate danger, please call your local emergency number. '
                'In the UK you can contact Samaritans 24/7 at 116 123 or text SHOUT to 85258.')
    return None
