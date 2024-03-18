from sentence_transformers import SentenceTransformer
import pandas as pd
import requests
import json
import numpy as np

# ollama pull nomic-embed-text

class embeddingsFactory:
    def __init__(self, urlbase, model):
        self.__model = model
        self.__urlbase = urlbase
        
    def createFromTXT(self, text):
        try: 
            url = self.__urlbase + "/embeddings"
            params = {"model": self.__model,
                      "prompt": text,
                      "keep_alive": 15}
            response = requests.post(url, json=params)
            if (response.status_code == 200):
                response_text = response.text
                data = json.loads(response_text)
                # returns a vector like this: {"embedding":[-1.5624014139175415,0.9712358713150024, ...]}
                final = {}
                final["text"] = text
                final["embedding"] = data["embedding"]
                # Build a vector like this: {'text': 'How many jobs Joe Biden wants to create ?', 'embedding': [-1.5624014139175415,  ...]}
                return final
            else:
                raise Exception("Error while reaching out the Web Service: {}", str(response.status_code, response.text))
            
        except Exception as e:
            print(e)
            return {}

    def createFromJSON(self, chunks):
        try: 
            final = []
            for chunk in chunks["chunks"]:
                final.append(self.createFromTXT(chunk))
            return final
        
        except Exception as e:
            print(e)
            return {}

    @staticmethod
    def createEmbeddingsFromTXT(text):
        try: 
            jsonInputs = {}
            jsonInputs["chunks"] = [text]
            answer = embeddingsFactory.createEmbeddingsFromJSON(jsonInputs)
            return answer
        except Exception as e:
            print(e)
            return {}
    
    @staticmethod
    def createEmbeddingsFromJSON(jsonChunks):
        try: 
            dfInput = pd.DataFrame(jsonChunks["chunks"], columns=["chunks"])
            encoder = SentenceTransformer("paraphrase-mpnet-base-v2")
            vect = encoder.encode(dfInput["chunks"])
            vectAndData = zip(dfInput["chunks"], vect)
            jsonOutput = {}
            for i, (chunk, vector) in enumerate(vectAndData):
                line = {}
                line["text"] = chunk
                line["embedding"] = vector
                jsonOutput[i] = line
            return jsonOutput
        except Exception as e:
            print(e)
            return {}