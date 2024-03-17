from sentence_transformers import SentenceTransformer
import pandas as pd

class embeddingsFactory:
       
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
                line["chunk"] = chunk
                line["vector"] = vector
                jsonOutput[i] = line
            return jsonOutput
        except Exception as e:
            print(e)
            return {}