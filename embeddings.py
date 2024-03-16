from sentence_transformers import SentenceTransformer
import pandas as pd

class embeddings:
    def createEmbeddingsFromTXT(self, text):
        try: 
            encoder = SentenceTransformer("paraphrase-mpnet-base-v2")
            vector = encoder.encode(text)
            answer = {}
            answer["text"] = text
            answer["vector"] = vector
            return answer
        except Exception as e:
            print(e)
            return {}

    def createEmbeddingsFromJSON(self, jsonChunks):
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