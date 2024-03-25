from sentence_transformers import SentenceTransformer
import pandas as pd
import requests
import json
import numpy as np

EMBEDDING_MODEL = "paraphrase-mpnet-base-v2"
# ollama pull nomic-embed-text

class embeddingsFactory:
    def __init__(self):
        try:
            self.__encoder = SentenceTransformer(EMBEDDING_MODEL)
        except:
            self.__encoder = None

    @property
    def encoder(self):
        return self.__encoder

    def createEmbeddingsFromTXT(self, text):
        try: 
            jsonInputs = {}
            jsonInputs["chunks"] = [text]
            textAndEmbedding = embeddingsFactory.createEmbeddingsFromJSON(jsonInputs)
            return textAndEmbedding
        except Exception as e:
            print(e)
            return {}
    
    def createEmbeddingsFromJSON(self, jsonChunks):
        try: 
            if (self.encoder == None):
                raise Exception ("Encoder not initialized")
            dfInput = pd.DataFrame(jsonChunks["chunks"], columns=["chunks"])
            vect = self.encoder.encode(dfInput["chunks"])
            vectAndData = zip(dfInput["chunks"], vect)
            textAndEmbeddings = {}
            for i, (chunk, vector) in enumerate(vectAndData):
                line = {}
                line["text"] = chunk
                line["embedding"] = vector
                textAndEmbeddings[i] = line
            return textAndEmbeddings
        except Exception as e:
            print(e)
            return {}