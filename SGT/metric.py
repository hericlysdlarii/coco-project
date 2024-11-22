import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# from nltk.translate.meteor_score import meteor_score, single_meteor_score
from rouge_score import rouge_scorer
# from pycocoevalcap.cider.cider import Cider

# from pymeteor import pymeteor
# from pycocoevalcap.spice.spice import Spice
# import spacy

class TextMetrics:
    def __init__(self):
        """
        Inicializa a classe para cálculo de métricas BLEU e METEOR.
        """
        self.smooth = SmoothingFunction().method1
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def calculate_bleu(self, reference, candidate):
        """
        Calcula a métrica BLEU entre uma referência e um candidato.
        :param reference: Texto de referência (string).
        :param candidate: Texto gerado (string).
        :return: BLEU Score (float).
        """
        reference_b = [reference.split()]  # Lista de listas
        candidate_b = candidate.split()    # Lista

        bleu = sentence_bleu(reference_b, candidate_b, smoothing_function=self.smooth)
        return bleu

    def calculate_rouge(self, reference, candidate):
        
        rouge = self.scorer.score(reference, candidate)
        return rouge

    # def calculate_spice(self, reference, candidate):
    #     """
    #     Calcula a métrica SPICE entre uma referência e um candidato.
    #     :param reference: Texto de referência (string).
    #     :param candidate: Texto gerado (string).
    #     :return: SPICE Score (float).
    #     """
    #     data = [{"image_id": 0, "captions": [reference]}]
    #     results = [{"image_id": 0, "caption": candidate}]
    #     spice_score = self.spice.compute_score(data, results)
    #     return spice_score[0]

    def evaluate(self, reference, candidate):
        bleu = self.calculate_bleu(reference, candidate),
        rouge = self.calculate_rouge(reference, candidate),
        return bleu, rouge