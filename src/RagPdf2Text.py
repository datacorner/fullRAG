import argparse
from utils.traceOut import traceOut
import utils.functions as F
import utils.CONST as C

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("-" + C.ARG_PDFFILE[0], help=C.ARG_PDFFILE[1], required=True)
        parser.add_argument("-" + C.ARG_TXTFILE[0], help=C.ARG_TXTFILE[1], required=True)
        args = vars(parser.parse_args())
        myTrace = traceOut(args)
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
        
    except Exception as e:
        F.wrapResponse(C.OUT_ERROR)
        F.wrapTrace(str(e))