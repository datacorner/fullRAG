import pandas as pd
import numpy as np
import faiss # pip install faiss-cpu (https://pypi.org/project/faiss-cpu/)
import pickle
import os

""" 
    Leverage Meta FAISS
"""
class FAISSWrapper:
    def __init__(self):
        self.index = None   # FAISS Index
        self.dfContent = pd.DataFrame(columns = ["text", "embedding"]) # real data that are indexed

    def save(self, filepath="./backup/", name="faissbackup"):
        """ Save the FAISS index and the data (chunks)
        Args:
            filepath (str, optional): _description_. Defaults to "./backup/".
            name (str, optional): _description_. Defaults to "faissbackup".
        """
        datafile = os.path.join(filepath, name + ".data")
        indexfile = os.path.join(filepath, name + ".index")
        with open(datafile, "wb") as f:
            pickle.dump(self.dfContent, f)
        faiss.write_index(self.index, indexfile)

    def load(self, filepath="./backup/", name="faissbackup"):
        """ Read the FAISS index and the data (chunks) saved previously

        Args:
            filepath (str, optional): _description_. Defaults to "./backup/".
            name (str, optional): _description_. Defaults to "faissbackup".
        """
        datafile = os.path.join(filepath, name + ".data")
        indexfile = os.path.join(filepath, name + ".index")
        with open(datafile, "rb") as f:
            self.dfContent = pickle.load(f)
        self.index = faiss.read_index(indexfile)
        
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
        
    def addToIndex(self, item):
        """ Index a new item
        Args:
            item (DataFrame): single embeddings to index
        """
        # Get source data and JSON -> DF
        dfNewContent = pd.DataFrame(item).T
        embeddings = [ np.asarray(v) for v in dfNewContent["embedding"] ]
        self.__addToIndexFlatL2(embeddings)
        # Concat the content with the existing DF
        self.dfContent = pd.concat([self.dfContent, dfNewContent])

    def __addToIndexFlatL2(self, embeddings):
        """
            Build a Flat L2 index
        """
        vout = self.__prepareEmbeddings(embeddings)
        self.index = faiss.IndexFlatL2(vout.shape[1])
        self.index.add(vout)

    def __prepareEmbeddings(self, vects):
        """ Prepare the embeddings for indexing
        Args:
            vects (array/embedding): vector
        Returns:
            array/embedding: vector prepared
        """
        vout =  np.asarray(vects)
        vout = vout.astype(np.float32) # Only support ndarray in 32 bits !
        faiss.normalize_L2(vout)
        return vout

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
        idx = "0" if type(list(prompt.keys())[0]) == "str" else 0
        vector = self.__prepareEmbeddings([ prompt[idx]["embedding"] ])
        # process the Similarity search
        k = self.index.ntotal
        distances, ann = self.index.search(vector, k=k)
        # Sort search results and return a DataFrame
        results = pd.DataFrame({'distances': distances[0], 'ann': ann[0]})
        self.dfContent.index = self.dfContent.index.astype(int)
        merge = pd.merge(results, self.dfContent, left_on='ann', right_index=True)
        return merge[:max]