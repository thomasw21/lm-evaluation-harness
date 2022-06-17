"""
Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank
https://aclanthology.org/D13-1170.pdf

The Stanford Sentiment Treebank is the first corpus with fully labeled parse trees that allows for a complete analysis of the compositional effects of sentiment in language.
 - SST2: Sentiment binary classification. Labels are rounded to either 0 or 1 to obtain a binary classification problem.

Homepage: https://nlp.stanford.edu/sentiment/index.html
"""
from lm_eval.base import Task, rf
from lm_eval.metrics import mean
from lm_eval.utils import general_detokenize

_CITATION = """
@inproceedings{socher-etal-2013-recursive,
    title = "Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank",
    author = "Socher, Richard  and
      Perelygin, Alex  and
      Wu, Jean  and
      Chuang, Jason  and
      Manning, Christopher D.  and
      Ng, Andrew  and
      Potts, Christopher",
    booktitle = "Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing",
    month = oct,
    year = "2013",
    address = "Seattle, Washington, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D13-1170",
    pages = "1631--1642",
}
"""


class StanfordSST2(Task):
    VERSION = 0
    DATASET_PATH = "sst"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            if self._training_docs is None:
                self._training_docs = list(self.dataset["train"])
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]

    def doc_to_text(self, doc):
        return "{}\nQuestion: Is this sentence positive or negative?\nAnswer:".format(
            general_detokenize(doc["sentence"]),
        )

    def doc_to_target(self, doc):
        label = int(doc["label"] > 0.5)
        return " {}".format({1: "positive", 0: "negative"}[label])

    def construct_requests(self, doc, ctx):
        ll_positive, _ = rf.loglikelihood(ctx, " positive")
        ll_negative, _ = rf.loglikelihood(ctx, " negative")
        return ll_positive, ll_negative

    def process_results(self, doc, results):
        ll_positive, ll_negative = results
        pred = ll_positive > ll_negative
        gold = doc["label"]
        return {"acc": pred == gold}

    def higher_is_better(self):
        return {"acc": True}

    def aggregation(self):
        return {"acc": mean}