import argparse
from elements.embeddingsFactory import embeddingsFactory
from utils.traceOut import traceOut
import utils.functions as functions

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
        parser.add_argument("-pdf", help="PDF file path", required=True)
        parser.add_argument("-faissname", help="FAISS Index reference", required=True)
        parser.add_argument("-path", help="Path to the FAISS Index", required=True)
        parser.add_argument("-chunk_size", help="Chunk Size", required=False, type=int, default=500)
        parser.add_argument("-chunk_overlap", help="Chunk Overlap", required=False, type=int, default=50)
        parser.add_argument("-separator", help="Separator", required=False, default=".")
        args = vars(parser.parse_args())
        myTrace = traceOut(args)
        myTrace.start()

        # 1 - Read the pdf content
        pdf = functions.readPDF(myTrace, args["pdf"])
        # 2 - Chunk document
        nb, chunks = functions.chunkContent(myTrace, pdf, args["separator"], args["chunk_size"], args["chunk_overlap"])
        embFactory = embeddingsFactory()
        # 3 - Chunks embeddings
        vChunks = functions.chunkEmbeddings(myTrace, embFactory, chunks)
        # 4 - Store embeddings in the index
        functions.FAISSStore(myTrace, vChunks, args["path"], args["faissname"])

        myTrace.stop()
        wrapTrace(myTrace.getFullJSON())
        
    except Exception as e:
        wrapResponse("ERROR")
        wrapTrace(str(e))