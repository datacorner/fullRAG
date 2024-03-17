import pandas as pd
import numpy as np
import faiss

class similaritySearch:
    def __init__(self):
        self.vectors = None
        self.index = None
        pass

    def loadChunks(self, jsonChunks):
        # Get source data and JSON -> DF
        self.dfContent = pd.DataFrame(jsonChunks).T

    def loadText(self, text):
        # Get source data and JSON -> DF
        vect = np.asarray(text[0]["vector"])
        vect = np.array([vect]).astype(np.float32)
        faiss.normalize_L2(vect)
        return vect

    @property
    def ready(self) -> bool: 
        try:
            # check if ready for searching for the NN
            return self.index.is_trained and not self.dfContent.empty
        except:
            return False

    def buildIndexFlatL2(self):
        # Build a FAISS index from the vectors
        vout =  np.asarray([ np.asarray(v) for v in self.dfContent["vector"] ])
        vout = vout.astype(np.float32) # Only support ndarray in 32 bits
        faiss.normalize_L2(vout)
        self.index = faiss.IndexFlatL2(vout.shape[1])
        self.addToIndexFlatL2(vout)

    def addToIndexFlatL2(self, _vectors):
        # Add vectors to an existing a FAISS index
        faiss.normalize_L2(_vectors)
        self.index.add(_vectors)

    def getNearest(self, _vector, _max):
        # k is set to the total number of vectors within the index
        # ann is the approximate nearest neighbour corresponding to those distances 
        # Similarity search
        k = self.index.ntotal
        distances, ann = self.index.search(_vector, k=k)
        # Sort search results
        results = pd.DataFrame({'distances': distances[0], 'ann': ann[0]})
        self.dfContent.index = self.dfContent.index.astype(int)
        merge = pd.merge(results, self.dfContent, left_on='ann', right_index=True)
        return merge[:_max]
