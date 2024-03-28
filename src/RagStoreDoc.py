import argparse
from elements.embeddingsFactory import embeddingsFactory
from utils.traceOut import traceOut
import utils.functions as F
import utils.CONST as C

def main():
    myTrace = traceOut(args)
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("-" + C.ARG_PDFFILE[0], help=C.ARG_PDFFILE[1], required=True)
        parser.add_argument("-" + C.ARG_FAISSNAME[0], help=C.ARG_FAISSNAME[1], required=True)
        parser.add_argument("-" + C.ARG_FAISSPATH[0], help=C.ARG_FAISSPATH[1], required=True)
        parser.add_argument("-" + C.ARG_CHUNKSIZE[0], help=C.ARG_CHUNKSIZE[1], required=False, type=int, default=500)
        parser.add_argument("-" + C.ARG_CHUNKOVAP[0], help=C.ARG_CHUNKOVAP[1], required=False, type=int, default=50)
        parser.add_argument("-" + C.ARG_SEP[0], help=C.ARG_SEP[1], required=False, default=".")
        args = vars(parser.parse_args())
        myTrace.start()

        # 1 - Read the pdf content
        pdf = F.readPDF(myTrace, args[C.ARG_PDFFILE[0]])
        # 2 - Chunk document
        nb, chunks = F.chunkContent(myTrace, pdf, args[C.ARG_SEP[0]], args[C.ARG_CHUNKSIZE[0]], args[C.ARG_CHUNKOVAP[0]])
        embFactory = embeddingsFactory()
        # 3 - Chunks embeddings
        vChunks = F.chunkEmbeddings(myTrace, embFactory, chunks)
        # 4 - Store embeddings in the index
        F.FAISSStore(myTrace,  vChunks,  C.ARG_FAISSPATH[0],  args[C.ARG_FAISSNAME[0]])

        myTrace.stop()
        F.wrapTrace(myTrace.getFullJSON())
        F.wrapResponse(C.OUT_SUCCESS)
        
    except Exception as e:
        F.wrapError(str(e))
        F.wrapResponse(C.OUT_ERROR)
        F.wrapTrace(myTrace.getFullJSON())
        
if __name__ == "__main__":
    main()