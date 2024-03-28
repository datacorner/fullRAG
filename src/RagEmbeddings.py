import argparse
from utils.traceOut import traceOut
import utils.functions as F
import utils.CONST as C
from elements.embeddingsFactory import embeddingsFactory

def getArg(arg, name):
    try:
        return arg[name]
    except:
        return C.NULLSTRING

def main():
    myTrace = traceOut()
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("-" + C.ARG_CHUNKS[0], help=C.ARG_CHUNKS[1], required=False, default=C.NULLSTRING)
        parser.add_argument("-" + C.ARG_PROMPT[0], help=C.ARG_PROMPT[1], required=False, default=C.NULLSTRING)
        parser.add_argument("-" + C.ARG_EMBEDDINGS[0], help=C.ARG_EMBEDDINGS[1], required=True)
        args = vars(parser.parse_args())
        myTrace.initialize(args)
        myTrace.start()

        # We must have a lit of chunks or a prompt, otherwise -> Exception
        chunks = getArg(args, C.ARG_CHUNKS[0])
        prompt = getArg(args, C.ARG_PROMPT[0])
        if (chunks == C.NULLSTRING and prompt == C.NULLSTRING or 
            chunks != C.NULLSTRING and prompt != C.NULLSTRING):
            raise Exception("A prompt or a list of chunks must be provided, but not both!")

        embFactory = embeddingsFactory()
        if (prompt != C.NULLSTRING):
            embeddings = F.textEmbeddings(myTrace, 
                                          embFactory, 
                                          args[C.ARG_PROMPT[0]])
        else:
            # Get the chunks first as list
            chunks = F.readJsonFromFile(args[C.ARG_CHUNKS[0]])
            embeddings = F.chunkEmbeddings(myTrace, 
                                           embFactory, 
                                           chunks)

        # Write the json in a file 
        if (not F.writeJsonToFile(args[C.ARG_EMBEDDINGS[0]], 
                                  embeddings)):
            raise Exception("Impossible to write the embeddings in a file")
        
        myTrace.stop()
        F.wrapTrace(myTrace.getFullJSON())
        F.wrapResponse(C.OUT_SUCCESS)
        F.wrapStatusOK()

    except Exception as e:
        F.wrapResponse(C.OUT_ERROR)
        F.wrapError(str(e))
        F.wrapTrace(myTrace.getFullJSON())
        F.wrapStatusERROR()
        
if __name__ == "__main__":
    main()