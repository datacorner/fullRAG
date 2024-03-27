# Description
Full RAG implementation with FAISS (for training)

# Installation

## via pip (command line)
pip install PyMuPDF  
pip install langchain  
pip install pandas  
pip install sentence_transformers  
pip install faiss-cpu  

## via requirements.txt (command line)
pip install -r requirements.txt

## Build setup
1) build/Modify the *.toml file
2) run py -m build
3) deploy / pyPI 
    twine upload --verbose dist/fullRAG-0.x.x-py3-none-any.whl
