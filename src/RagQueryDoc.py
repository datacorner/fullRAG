import argparse
from elements.embeddingsFactory import embeddingsFactory
from utils.traceOut import traceOut
import utils.functions as functions
from elements.similaritySearchEngine import similaritySearchEngine
import utils.CONST as C

def wrapResponse(response):
    """ Wrap the response between 2 XML tags to avoid a mix with the command line output/errors
    Args:
        response (_type_): response wrapped
    """
    print("<response>" + response + "</response>")

def wrapTrace(response):
    print("<log>" + response + "</log>")

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("-prompt", help="Prompt to send to LLAMA2", required=True)
        parser.add_argument("-pdf", help="PDF file path", required=True)
        parser.add_argument("-temperature", help="LLM Temperature", required=False, type=float, default=0.9) # float(self.temperature.replace(",", "."))
        parser.add_argument("-chunk_size", help="Chunk Size", required=False, type=int, default=500)
        parser.add_argument("-chunk_overlap", help="Chunk Overlap", required=False, type=int, default=50)
        parser.add_argument("-nearest", help="Faiss Number of nearest to gather for prompting", required=False, type=int, default=3)
        parser.add_argument("-separator", help="Separator", required=False, default=".")
        parser.add_argument("-model", help="Ollama Model installed", required=False, default="tinydolphin")
        parser.add_argument("-urlbase", help="URL for Ollama API (default localhost)", required=False, default="http://localhost:11434/api")
        args = vars(parser.parse_args())
        myTrace = traceOut(args)
        myTrace.start()

        # 1 - Read the pdf content
        pdf = functions.readPDF(myTrace, args["pdf"])
        # 2 - Chunk document
        nb, chunks = functions.chunkContent(myTrace, pdf, args["separator"], args["chunk_size"], args["chunk_overlap"])
        # 3 - Text embeddings
        embFactory = embeddingsFactory()
        vPrompt = functions.textEmbeddings(myTrace, embFactory, args["prompt"])
        # 4 - Chunks embeddings
        vChunks = functions.chunkEmbeddings(myTrace, embFactory, chunks)
        # 5 - Index the chunks
        myfaiss = similaritySearchEngine()
        functions.FAISSaddToIndex(myTrace, myfaiss, vChunks)
        # 6 - Similarity Search
        similars = functions.FAISSSearch(myTrace, myfaiss, args["nearest"], vPrompt)
        # 7 - Build prompt
        customPrompt = functions.buildPrompt(myTrace, args["prompt"], similars["text"])
        # 8 - Ask to the LLM ...
        resp = functions.promptLLM(myTrace, customPrompt, args["urlbase"], args["model"], args["temperature"])
    
        myTrace.stop()
        wrapTrace(myTrace.getFullJSON())
        wrapResponse(resp)
        
    except Exception as e:
        wrapResponse(C.OUT_ERROR)
        wrapTrace(str(e))