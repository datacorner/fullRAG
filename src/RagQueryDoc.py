import argparse
from elements.embeddingsFactory import embeddingsFactory
from utils.traceOut import traceOut
import utils.functions as F
from elements.FAISSWrapper import FAISSWrapper
import utils.CONST as C

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("-" + C.ARG_PROMPT[0], help=C.ARG_PROMPT[1], required=True)
        parser.add_argument("-" + C.ARG_PDFFILE[0], help=C.ARG_PDFFILE[1], required=True)
        parser.add_argument("-" + C.ARG_TEMP[0], help=C.ARG_TEMP[1], required=False, type=float, default=0.9) # float(self.temperature.replace(",", "."))
        parser.add_argument("-" + C.ARG_CHUNKSIZE[0], help=C.ARG_CHUNKSIZE[1], required=False, type=int, default=500)
        parser.add_argument("-" + C.ARG_CHUNKOVAP[0], help=C.ARG_CHUNKOVAP[1], required=False, type=int, default=50)
        parser.add_argument("-" + C.ARG_SEP[0], help=C.ARG_SEP[1], required=False, default=".")
        parser.add_argument("-" + C.ARG_NEAREST[0], help=C.ARG_NEAREST[1], required=False, type=int, default=3)
        parser.add_argument("-" + C.ARG_MODEL[0], help=C.ARG_MODEL[1], required=False, default="tinydolphin")
        parser.add_argument("-" + C.ARG_URL[0], help=C.ARG_URL[1], required=False, default="http://localhost:11434/api")
        args = vars(parser.parse_args())
        myTrace = traceOut(args)
        myTrace.start()

        # 1 - Read the pdf content
        pdf = F.readPDF(myTrace, args[C.ARG_PDFFILE[0]])
        # 2 - Chunk document
        nb, chunks = F.chunkContent(myTrace, pdf, args[C.ARG_SEP[0]], args[C.ARG_CHUNKSIZE[0]], args[C.ARG_CHUNKOVAP[0]])
        # 3 - Text embeddings
        embFactory = embeddingsFactory()
        vPrompt = F.textEmbeddings(myTrace, embFactory, args[C.ARG_PROMPT[0]])
        # 4 - Chunks embeddings
        vChunks = F.chunkEmbeddings(myTrace, embFactory, chunks)
        # 5 - Index the chunks
        myfaiss = FAISSWrapper()
        F.FAISSaddToIndex(myTrace, myfaiss, vChunks)
        # 6 - Similarity Search
        similars = F.FAISSSearch(myTrace, myfaiss, args[C.ARG_NEAREST[0]], vPrompt)
        # 7 - Build prompt
        customPrompt = F.buildPrompt(myTrace, args[C.ARG_PROMPT[0]], similars["text"])
        # 8 - Ask to the LLM ...
        resp = F.promptLLM(myTrace, customPrompt, args[C.ARG_URL[0]], args[C.ARG_MODEL[0]], args[C.ARG_TEMP[0]])
    
        myTrace.stop()
        F.wrapTrace(myTrace.getFullJSON())
        F.wrapResponse(resp)
        
    except Exception as e:
        F.wrapResponse(C.OUT_ERROR)
        F.wrapTrace(str(e))