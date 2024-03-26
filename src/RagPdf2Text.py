import argparse
from utils.traceOut import traceOut
import utils.functions as functions
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
        parser.add_argument("-pdf", help="PDF Input file path", required=True)
        parser.add_argument("-txt", help="TXT Output file path", required=True)
        args = vars(parser.parse_args())
        myTrace = traceOut(args)
        myTrace.start()

        # 1 - Read the pdf content
        pdf = functions.readPDF(myTrace, args["pdf"])
        if not functions.writeToFile(args["txt"], pdf.content):
            raise Exception("Impossible to write the content into the file")
        myTrace.stop()
        wrapTrace(myTrace.getFullJSON())
        wrapResponse(C.OUT_SUCCESS)
        
    except Exception as e:
        wrapResponse(C.OUT_ERROR)
        wrapTrace(str(e))