from evaluate import load

rouge = load("rouge", module_type='metric')
bleu = load("bleu", module_type='metric')
meteor = load("meteor", module_type="metric")


def compute_rouge(pred, target):
    # compute rouge score for a prediction and a target
    # pred: predicted sentence
    # target: target sentence

    rouge_score = rouge.compute(references=target, predictions=pred)
    return rouge_score['rougeL']


def compute_bleu(pred, target):
    # compute bleu score for a prediction and a target
    # pred: predicted sentence
    # target: target sentence

    bleu_score = bleu.compute(references=target, predictions=pred)
    return bleu_score['bleu']


def compute_meteor(pred, target):
    # compute meteor score for a prediction and a target
    # pred: predicted sentence
    # target: target sentence

    meteor_score = meteor.compute(references=target, predictions=pred)
    return meteor_score['meteor']


def compute_roberta_eval(pred, target):
    pass


def get_metric(metric_name):
    # get metric function based on metric name
    # metric_name: name of the metric

    if metric_name == 'rouge':
        return compute_rouge
    elif metric_name == 'bleu':
        return compute_bleu
    elif metric_name == 'meteor':
        return compute_meteor
    else:
        raise ValueError(f'Invalid metric name. Got {metric_name}, expected one of [rouge, bleu, meteor, bleurt]')
