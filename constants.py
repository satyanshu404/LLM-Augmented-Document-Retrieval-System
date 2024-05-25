from dataclasses import dataclass
from os import CLD_CONTINUED

@dataclass
class Constants:
    TEXT_MODEL_CHECKPOINT: str = "satyanshu404/gpt2-finetuned-justification-v5"
    CLASSIFICATIN_MODEL_CHECKPOINT: str = "satyanshu404/relevance-classification-v1"
    HF_KEY = ""
    TEXT_PREFIX = "For the given query, document and document's relevancy, Provide a justification for the document's relevance to the given query."
    CLASSI_PREFIX = "For the given query and a document, find document's relevance to the given query."
