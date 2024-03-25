PROMPT_TEMPLATE = "Question: {prompt}\n \
Please answer the question based on the informations listed below: \
{context}"
ITEM_CONTEXT_TEMPLATE_LINE = "Context {i}: {contextItem}"

class prompt:
    def __init__(self, myquestion, mycontext):
        self.__question = myquestion
        self.__context = mycontext  # Pandas Serie
        self.__template = PROMPT_TEMPLATE

    @property
    def template(self):
        return self.__template
    @template.setter
    def template(self, t):
        self.__template = t
        
    @property
    def question(self):
        return self.__question
    @question.setter
    def question(self, q):
        self.__question = q
    
    @property
    def context(self):
        return self.__context
    @context.setter
    def context(self, q):
        self.__context = q
    
    def build(self):
        try: 
            itemContext = ""
            for i, item in self.context.items():
                itemContext = itemContext + ITEM_CONTEXT_TEMPLATE_LINE.format(i=i, 
                                                                            contextItem=item) + "\n"
            return self.template.format(prompt=self.question, 
                                        context=itemContext)
        except:
            return ""