import fitz # pip install PyMuPDF
from langchain.text_splitter import CharacterTextSplitter

class document:
    def __init__(self, __filepath):
        self.__filepath = __filepath
        self.__content = ""
        
    @property
    def filepath(self) -> str: 
        return self.__filepath
    
    @property
    def content(self) -> str: 
        return self.__content
    
    def getContentFromPDF(self, fromPage=0, toPage=0, heightToRemove=0) -> bool:
        try:
            reader = fitz.open(self.__filepath)
            for numPage, page in enumerate(reader): # iterate the document pages
                toPage = len(reader) if (toPage == 0) else toPage 
                if (numPage+1 >= fromPage and numPage+1 <= toPage):
                    pageBox = page.artbox
                    rect = fitz.Rect(pageBox[0], 
                                    pageBox[1] + heightToRemove, 
                                    pageBox[2], 
                                    pageBox[3] - heightToRemove)
                    self.__content = self.__content + page.get_textbox(rect) # get plain text encoded as UTF-8
            return True
        except Exception as e:
            print(e)
            return False
        
    def chunk(self, separator, chunk_size, chunk_overlap):
        try: 
            text_splitter = CharacterTextSplitter(separator = separator, 
                                                chunk_size = chunk_size, 
                                                chunk_overlap = chunk_overlap, 
                                                length_function = len, 
                                                is_separator_regex = False)
            docs = text_splitter.create_documents([self.__content])
            nbChunks = len(docs)
            jsonInputs = {}
            jsonInputs["chunks"] = [ x.page_content for x in docs ] 
            # return a JSON like this : {'chunks': ['Transcript of ...', ...] }
            return nbChunks, jsonInputs
        except Exception as e:
            print(e)
            return -1, {}