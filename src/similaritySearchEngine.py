import pandas as pd
import numpy as np
import faiss
import pickle

""" 
    Leverage Meta FAISS
"""
class similaritySearchEngine:
    def __init__(self):
        self.index = None   # FAISS Index
        self.dfContent = pd.DataFrame() # real data that are indexed

    def save(self, filepath="./backup/", name="faissbackup"):
        """ Save the FAISS index and the data (chunks)
        Args:
            filepath (str, optional): _description_. Defaults to "./backup/".
            name (str, optional): _description_. Defaults to "faissbackup".
        """
        datafile = filepath + name + ".data"
        indexfile = filepath + name + ".index"
        with open(datafile, "wb") as f:
            pickle.dump(self.dfContent, f)
        faiss.write_index(self.index, indexfile)

    def read(self, filepath="./backup/", name="faissbackup"):
        """ Read the FAISS index and the data (chunks) saved previously

        Args:
            filepath (str, optional): _description_. Defaults to "./backup/".
            name (str, optional): _description_. Defaults to "faissbackup".
        """
        datafile = filepath + name + ".data"
        indexfile = filepath + name + ".index"
        with open(datafile, "rb") as f:
            self.dfContent = pickle.load(f)
        self.index = faiss.read_index(indexfile)

    def addToIndex(self, item):
        """ Index a new item
        Args:
            item (json): single embeddings to index
        """
        # Get source data and JSON -> DF
        self.dfContent = pd.DataFrame(item).T
        self.__buildIndexFlatL2()

    @property
    def ready(self) -> bool:
        """check if ready for searching for the NN
        Returns:
            bool: True if index ready
        """
        try:
            return self.index.is_trained #and not self.dfContent.empty
        except:
            return False

    def __buildIndexFlatL2(self):
        """
            Build a Flat L2 index
        """
        vout =  np.asarray([ np.asarray(v) for v in self.dfContent["embedding"] ])
        vout = vout.astype(np.float32) # Only support ndarray in 32 bits
        faiss.normalize_L2(vout)
        self.index = faiss.IndexFlatL2(vout.shape[1])
        self.__addToIndexFlatL2(vout)

    def __addToIndexFlatL2(self, vector):
        """ Add vectors to an existing a FAISS index
        Args:
            _vectors (_type_): _description_
        """
        faiss.normalize_L2(vector)
        self.index.add(vector)

    def getNearest(self, prompt, max):
        """ Process the similarity search on the existing FAISS index (and the given prompt)
                --> k is set to the total number of vectors within the index
                --> ann is the approximate nearest neighbour corresponding to those distances
        Args:
            prompt (json): Prompt's embeddings
            max (_type_): Nb of nearest to return
        Returns:
            DataFrame: List of the most nearest neighbors
        """
        # Get prompt vector only and normalize it
        vector = np.asarray(prompt[0]["embedding"])
        vector = np.array([vector]).astype(np.float32)
        faiss.normalize_L2(vector)
        # process the Similarity search
        k = self.index.ntotal
        distances, ann = self.index.search(vector, k=k)
        # Sort search results and return a DataFrame
        results = pd.DataFrame({'distances': distances[0], 'ann': ann[0]})
        self.dfContent.index = self.dfContent.index.astype(int)
        merge = pd.merge(results, self.dfContent, left_on='ann', right_index=True)
        return merge[:max]
                                                                           