"""Post-processing of generated programs and answers (code-fence extraction, answer parsing).

Pure text utilities; no saturn.perception/vlm/serving imports.
"""
import re
import torch

from saturn.log import get_logger

log = get_logger(__name__)


def unified_postprocess(prediction):
    contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
                    "couldnt": "couldn't", "couldn'tve": "couldn't've", "couldnt've": "couldn't've",
                    "didnt": "didn't","doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't",
                    "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't",
                    "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've", "hes": "he's", "howd": "how'd",
                    "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im": "I'm",
                    "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've",
                    "itll": "it'll", "let's": "let's", "maam": "ma'am", "mightnt": "mightn't",
                    "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
                    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've",
                    "oclock": "o'clock", "oughtnt": "oughtn't", "ow's'at": "'ow's'at", "'ows'at": "'ow's'at",
                    "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've",
                    "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't",
                    "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", "somebody'd": "somebodyd",
                    "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've",
                    "somebodyll": "somebody'll", "somebodys": "somebody's", "someoned": "someone'd",
                    "someoned've": "someone'd've", "someone'dve": "someone'd've", "someonell": "someone'll",
                    "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've",
                    "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's",
                    "thered": "there'd", "thered've": "there'd've", "there'dve": "there'd've",
                    "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've",
                    "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've",
                    "twas": "'twas", "wasnt": "wasn't", "wed've": "we'd've", "we'dve": "we'd've",
                    "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're",
                    "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd",
                    "wheres": "where's", "whereve": "where've", "whod": "who'd", "whod've": "who'd've",
                    "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've",
                    "whyll": "why'll", "whyre": "why're", "whys": "why's", "wont": "won't",
                    "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
                    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll",
                    "yall'd've": "y'all'd've", "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've",
                    "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", "youll": "you'll",
                    "youre": "you're", "youve": "you've"}
    manual_map = {
        'none': '0', 'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10'
    }
    
    articles = ['a', 'an', 'the']
    punct = [';', r"/", '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_',
             '-', '>', '<', '@', '`', ',', '?', '!']
    period_strip = re.compile(r"(?!<=\d)(\.)(?!\d)")
    comma_strip = re.compile(r"(\d)(\,)(\d)")

    # 1. General preprocessing
    try:
        if type(prediction).__name__ == 'ProbabilisticTensor':
            prediction = prediction.arg

        if isinstance(prediction, list):
            prediction = prediction[0] if len(prediction) > 0 else "no"

        if isinstance(prediction, torch.Tensor):
            prediction = prediction.item()

        if prediction is None:
            prediction = "no"

        if isinstance(prediction, bool):
            prediction = "yes" if prediction else "no"
        elif isinstance(prediction, int):
            prediction = str(prediction)
            log.warning("No answer is a number, so this will be wrong")
        elif isinstance(prediction, str):
            prediction = prediction.strip()
            if prediction.endswith('s') and prediction.strip().lower() not in ['yes', 'no']:
                prediction = prediction[:-1]
    except Exception:
        prediction = str(prediction)

    prediction = str(prediction)
    prediction = prediction.replace('\n', ' ').replace('\t', ' ').strip().lower()
    if prediction == 'true':
        prediction = 'yes'
    elif prediction == 'false':
        prediction = 'no'

    # 2. processPunctuation logic
    out_text = prediction
    for p in punct:
        if (p + ' ' in out_text or ' ' + p in out_text) or re.search(comma_strip, out_text) is not None:
            out_text = out_text.replace(p, '')
        else:
            out_text = out_text.replace(p, ' ')
    out_text = period_strip.sub("", out_text)

    # 3. processDigitArticle logic
    temp_words = out_text.lower().split()
    out_words = []
    for word in temp_words:
        word = manual_map.get(word, word)
        if word not in articles:
            out_words.append(word)
    out_words = [contractions.get(word, word) for word in out_words]
    
    return ' '.join(out_words)

