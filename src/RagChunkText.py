import argparse
from utils.traceOut import traceOut
import utils.functions as F
import utils.CONST as C
from elements.document import document

def main():
    myTrace = traceOut()
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("-" + C.ARG_TXTFILE[0], help=C.ARG_TXTFILE[1], required=True)
        parser.add_argument("-" + C.ARG_CHUNKS[0], help=C.ARG_CHUNKS[1], required=True)
        parser.add_argument("-" + C.ARG_CHUNKSIZE[0], help=C.ARG_CHUNKSIZE[1], required=False, type=int, default=500)
        parser.add_argument("-" + C.ARG_CHUNKOVAP[0], help=C.ARG_CHUNKOVAP[1], required=False, type=int, default=50)
        parser.add_argument("-" + C.ARG_SEP[0], help=C.ARG_SEP[1], required=False, default=".")
        args = vars(parser.parse_args())
        myTrace.initialize(args)
        myTrace.start()

        # Read the Text file first
        nb = -1
        doc = document(args[C.ARG_TXTFILE[0]])
        
        # Document chunking
        if (doc.getContentFromTXT()):
            nb, chunks = F.chunkContent(myTrace, 
                                        doc, 
                                        args[C.ARG_SEP[0]], 
                                        args[C.ARG_CHUNKSIZE[0]], 
                                        args[C.ARG_CHUNKOVAP[0]])
    
        # Write the json in a file 
        if (not F.writeJsonToFile(args[C.ARG_CHUNKS[0]], 
                                  chunks)):
            raise Exception("Impossible to write the chunks in a file")

        myTrace.stop()
        F.wrapTrace(myTrace.getFullJSON())
        F.wrapResponse(str(nb))
        F.wrapStatusOK()

    except Exception as e:
        F.wrapResponse(C.OUT_ERROR)
        F.wrapError(str(e))
        F.wrapTrace(myTrace.getFullJSON())
        F.wrapStatusERROR()
        
if __name__ == "__main__":
    main()