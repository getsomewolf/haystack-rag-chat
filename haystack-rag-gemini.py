import os
from pathlib import Path
from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.builders import PromptBuilder
# Importar o Gerador Ollama
from haystack_integrations.components.generators.ollama import OllamaGenerator # Use OllamaChatGenerator se preferir formato chat

# --- Configuração Inicial ---
pdf_dir = Path(".")
pdf_files = list(pdf_dir.glob("*.pdf"))

if not pdf_files:
    print(f"Nenhum arquivo PDF encontrado em '{pdf_dir}'. Crie a pasta e adicione PDFs.")
    exit()

print(f"Arquivos PDF encontrados: {[str(p) for p in pdf_files]}")

# --- 1. Inicializar o Document Store ---
print("Inicializando Document Store...")
document_store = InMemoryDocumentStore()

# --- 2. Pipeline de Indexação (Mesmo de antes) ---
print("Configurando pipeline de indexação...")
file_converter = PyPDFToDocument()
document_splitter = DocumentSplitter(split_by="word", split_length=150, split_overlap=50)
# Modelo de embedding leve e eficaz
doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
doc_writer = DocumentWriter(document_store=document_store)

indexing_pipeline = Pipeline()
indexing_pipeline.add_component("converter", file_converter)
indexing_pipeline.add_component("splitter", document_splitter)
indexing_pipeline.add_component("embedder", doc_embedder)
indexing_pipeline.add_component("writer", doc_writer)

indexing_pipeline.connect("converter.documents", "splitter.documents")
indexing_pipeline.connect("splitter.documents", "embedder.documents")
indexing_pipeline.connect("embedder.documents", "writer.documents")

print("Iniciando indexação dos PDFs...")
indexing_pipeline.run({"converter": {"sources": [str(p) for p in pdf_files]}})
print(f"Indexação concluída. {document_store.count_documents()} chunks armazenados.")

# --- 3. Pipeline de Query (RAG com Ollama) ---
print("Configurando pipeline de RAG (Query) com Ollama...")
text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
retriever = InMemoryEmbeddingRetriever(document_store=document_store, top_k=3)

# Template do Prompt - Ajustado ligeiramente para modelos Instruct
prompt_template = """
Use APENAS os seguintes trechos de documentos para responder à pergunta. Não use nenhum conhecimento prévio.
Se a resposta não estiver nos documentos, diga explicitamente "Não encontrei a resposta nos documentos fornecidos.".

Documentos:
{% for doc in documents %}
  Trecho: {{ doc.content }}
{% endfor %}

Pergunta: {{ query }}
Resposta:
"""
prompt_builder = PromptBuilder(template=prompt_template)

# Gerador de Resposta (LLM) - Usando Ollama
# Certifique-se que o modelo ('mistral', 'llama3', 'phi3') foi baixado com 'ollama pull'
# e que o servidor Ollama está rodando.
llm = OllamaGenerator(
    model="mistral", # Ou 'llama3', 'phi3', etc.
    url="http://localhost:11434/api/generate", # URL padrão do Ollama
    generation_kwargs={ # Opcional: ajuste parâmetros de geração
        "num_predict": 150, # Máximo de tokens na resposta
        "temperature": 0.3, # Mais baixo = mais determinístico
        # "top_p": 0.9,
        # "stop": ["\n", "Pergunta:"] # Palavras/tokens para parar a geração
    }
)

# Construir o pipeline de RAG/Query
rag_pipeline = Pipeline()
rag_pipeline.add_component("text_embedder", text_embedder)
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("prompt_builder", prompt_builder)
rag_pipeline.add_component("llm", llm)

# Conectar os componentes
rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder.prompt", "llm.prompt")

# --- 4. Fazer uma Pergunta ---
while True:
    try:
        query = input("\nFaça sua pergunta (ou digite 'sair' para terminar): ")
        if query.lower() == 'sair':
            break
        if not query:
            continue

        print("\nProcessando pergunta com Ollama...")
        # Executar o pipeline RAG
        results = rag_pipeline.run(
            {
                "text_embedder": {"text": query},
                "prompt_builder": {"query": query}
            }
        )

        # Imprimir a resposta gerada pelo LLM Ollama
        # A resposta está dentro da chave 'replies' no resultado do OllamaGenerator
        if results["llm"]["replies"]:
            resposta_gerada = results["llm"]["replies"][0]
            print("\nResposta:")
            print(resposta_gerada.strip()) # .strip() para remover espaços extras
        else:
            print("\nO modelo não gerou uma resposta.")

    except Exception as e:
        print(f"Ocorreu um erro: {e}")
        import traceback
        traceback.print_exc() # Imprime mais detalhes do erro

print("\nEncerrando.")