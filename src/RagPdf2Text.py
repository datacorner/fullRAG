import argparse
from utils.traceOut import traceOut
import utils.functions as F
import utils.CONST as C

def main():
    myTrace = traceOut()
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("-" + C.ARG_PDFFILE[0], help=C.ARG_PDFFILE[1], required=True)
        parser.add_argument("-" + C.ARG_TXTFILE[0], help=C.ARG_TXTFILE[1], required=True)
        args = vars(parser.parse_args())
        myTrace.initialize(args)
        myTrace.start()

        # 1 - Read the pdf content
        pdf = F.readPDF(myTrace, 
                        args[C.ARG_PDFFILE[0]])
        if not F.writeToFile(args[C.ARG_TXTFILE[0]], 
                             pdf.content):
            raise Exception("Impossible to write the content into the file")
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