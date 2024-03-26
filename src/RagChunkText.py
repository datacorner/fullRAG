import argparse
from utils.traceOut import traceOut
import utils.functions as functions
import utils.CONST as C
from elements.document import document

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
        parser.add_argument("-txt", help="Input Text file path", required=True)
        parser.add_argument("-out", help="Output JSON file path for the chunks", required=True)
        parser.add_argument("-chunk_size", help="Chunk Size", required=False, type=int, default=500)
        parser.add_argument("-chunk_overlap", help="Chunk Overlap", required=False, type=int, default=50)
        parser.add_argument("-separator", help="Separator", required=False, default=".")
        args = vars(parser.parse_args())
        myTrace = traceOut(args)
        myTrace.start()

        # Read the Text file first
        nb = -1
        doc = document(args["txt"])
        
        # Document chunking
        if (doc.getContentFromTXT()):
            nb, chunks = functions.chunkContent(myTrace, doc, args["separator"], args["chunk_size"], args["chunk_overlap"])
    
        # Write the json in a file 
        if (not functions.writeJsonToFile(args["out"], chunks)):
            raise Exception("Impossible to write the chunks in a file")

        myTrace.stop()
        wrapTrace(myTrace.getFullJSON())
        wrapResponse(str(nb))
        
    except Exception as e:
        wrapResponse(C.OUT_ERROR)
        wrapTrace(str(e))