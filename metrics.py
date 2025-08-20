from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def exact_match(pred, answers):
    # Normalize strings: lowercase + strip spaces
    pred_norm = pred.lower().strip()
    answers_norm = [a.lower().strip() for a in answers]
    return int(pred_norm in answers_norm)

def bleu(pred, answers):
    
    refs = [a.split() for a in answers]   # multiple references
    smoothie = SmoothingFunction().method3
    return sentence_bleu(refs, pred.split(), smoothing_function=smoothie)