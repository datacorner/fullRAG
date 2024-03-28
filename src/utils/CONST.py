NULLSTRING = ""

# Output status
OUT_ERROR = "ERROR"
OUT_SUCCESS = "SUCCESS"

# Output TAGs
TAG_O_RESPONSE = "<response>"
TAG_C_RESPONSE = "</response>"
TAG_O_LOG = "<log>"
TAG_C_LOG = "</log>"
TAG_O_ERROR = "<error>"
TAG_C_ERROR = "</error>"

# Command line arguments
ARG_PROMPT = ["prompt", "Prompt to send to the LLM"]
ARG_TEMP = ["temperature", "LLM Temperature parameter, by defaul 0.9"]
ARG_PDFFILE = ["pdf", "PDF file path"]
ARG_TXTFILE = ["txt", "Text file path"]
ARG_CHUNKS = ["chunks", "JSON file path which contains the chunks"]
ARG_FAISSNAME = ["faissname", "FAISS Index reference name"]
ARG_FAISSPATH = ["faisspath", "FAISS Index reference path (FAISS index and data storage)"]
ARG_CHUNKSIZE = ["chunk_size", "Chunk Size"]
ARG_CHUNKOVAP = ["chunk_overlap", "Chunk Overlap"]
ARG_SEP = ["sep", "Separator for chunking"]
ARG_NEAREST = ["nearest", "Faiss Number of nearest chunks to gather for prompting"]
ARG_MODEL = ["model", "Ollama Model installed"]
ARG_URL = ["urlbase", "URL for Ollama API (default localhost)"]
ARG_EMBEDDINGS = ["embeddings", "JSON file path which contains the data and embeddings"]
ARG_EMBEDDINGS_PT = ["embprompt", "JSON file path which contains the data and embeddings for the prompt"]
ARG_EMBEDDINGS_CK = ["embchunks", "JSON file path which contains the data and embeddings for the chunks"]
ARG_FAISSACTION = ["action", "Action to execute on the FAISS Engine (store/indexsearch/memsearch)"]
ARG_FAISSACTION_VALMSEARCH = "memsearch"
ARG_FAISSACTION_VALISEARCH = "indexsearch"
ARG_FAISSACTION_VALSTORE = "store"
ARG_NEAREST = ["nfile", "JSON file path which contains the nearest chunks/texts"]