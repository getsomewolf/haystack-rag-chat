# Importações
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import PDFToTextConverter, PreProcessor, BM25Retriever, FARMReader
from haystack.pipelines import Pipeline, ExtractiveQAPipeline
from haystack import Document
from haystack.utils import print_answers

# Passo 1: Configurar o Document Store
document_store = InMemoryDocumentStore()

# Passo 2: Converter PDF para texto
pdf_converter = PDFToTextConverter(remove_numeric_tables=True)
pdf_path = "Palhano_da_Silva_Ribeiro_Lucas_CV.pdf"
text = pdf_converter.convert(file_path=pdf_path)

# Passo 3: Criar e pré-processar o documento
doc = Document(content=text[0].text)
preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=True,
    clean_header_footer=True,
    split_by="word",
    split_length=100,
    split_overlap=10,
)
docs = preprocessor.process([doc])
document_store.write_documents(docs)

# Passo 4: Configurar Retriever e Reader
retriever = BM25Retriever(document_store=document_store)
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")

# Passo 5: Criar o pipeline de QA
pipe = ExtractiveQAPipeline(reader, retriever)

# Passo 6: Fazer uma pergunta
query = "Quando que o Lucas aprendeu a jogar xadrez e tornou sua paixão?"
prediction = pipe.run(query=query, params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}})
print_answers(prediction, details="minimum")