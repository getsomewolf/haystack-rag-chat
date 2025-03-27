import os
from pathlib import Path
import requests
import time
import sys
import threading


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
# Configurações e constantes
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_TIMEOUT = 180  # Timeout em segundos
MODEL_NAME = "mistral"  # Ou 'llama3', 'phi3', etc.

# Verificação do servidor Ollama
def check_ollama_server():
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            models_list = [m.get("name") for m in models]
            
            if not models_list:
                print(f"⚠️ Nenhum modelo encontrado no servidor Ollama.")
                return False
                
            if MODEL_NAME not in [m.split(':')[0] for m in models_list]:
                print(f"⚠️ Modelo '{MODEL_NAME}' não encontrado. Modelos disponíveis: {models_list}")
                print(f"Execute 'ollama pull {MODEL_NAME}' para baixá-lo")
                return False
                
            print(f"✅ Servidor Ollama encontrado com modelo '{MODEL_NAME}' disponível")
            return True
    except requests.exceptions.ConnectionError:
        print(f"❌ Servidor Ollama não encontrado em {OLLAMA_BASE_URL}")
        print("   Certifique-se que o servidor está rodando com 'ollama serve'")
    except requests.exceptions.Timeout:
        print(f"❌ Timeout ao conectar com servidor Ollama")
    except Exception as e:
        print(f"❌ Erro ao verificar servidor Ollama: {e}")
    return False

# --- Carregamento de PDFs ---
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

# Verificar conexão com Ollama antes de prosseguir
if not check_ollama_server():
    print("\nVerifique se o servidor Ollama está rodando.")
    print("Execute 'ollama serve' e 'ollama pull mistral' antes de continuar.")
    exit(1)

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

# Gerador de Resposta (LLM) - Usando Ollama com timeout aumentado
llm = OllamaGenerator(
    model=MODEL_NAME,
    url=OLLAMA_BASE_URL,
    timeout=OLLAMA_TIMEOUT,
    generation_kwargs={
        "num_predict": 150,
        "temperature": 0.3,
    }
)

# Separar em dois pipelines: um de recuperação e um de geração
retrieval_pipeline = Pipeline()
retrieval_pipeline.add_component("text_embedder", text_embedder)
retrieval_pipeline.add_component("retriever", retriever)
retrieval_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

generation_pipeline = Pipeline()
generation_pipeline.add_component("prompt_builder", prompt_builder)
generation_pipeline.add_component("llm", llm)
generation_pipeline.connect("prompt_builder.prompt", "llm.prompt")

# --- 4. Fazer uma Pergunta ---
while True:
    try:
        query = input("\nFaça sua pergunta (ou digite 'sair' para terminar): ")
        if query.lower() == 'sair':
            break
        if not query:
            continue

        print("\n🔎 Buscando documentos relevantes...")
        start_time = time.time()
        
        def show_processing_animation():
            elapsed_time = time.time() - start_time
            remaining = max(0, OLLAMA_TIMEOUT - elapsed_time)
            sys.stdout.write(f"\r⏳ Gerando resposta... {elapsed_time:.1f}s decorridos (timeout: {OLLAMA_TIMEOUT}s, restante: {remaining:.1f}s)")
            sys.stdout.flush()
        
        # Buscar documentos primeiro (mais rápido)
        retrieval_results = retrieval_pipeline.run(
            {
                "text_embedder": {"text": query},
            }
        )
        
        print(f"✅ {len(retrieval_results['retriever']['documents'])} documentos encontrados.")
        print("\n🧠 Gerando resposta com o modelo...")
        
        # Iniciar animação de processamento
        try:
            animation_thread = None
            stop_animation = threading.Event()
            
            # Função contínua para a animação
            def animation_loop():
                while not stop_animation.is_set():
                    show_processing_animation()
                    time.sleep(0.5)
                # Limpar a linha quando a animação parar
                sys.stdout.write("\r" + " " * 100 + "\r")
                sys.stdout.flush()
            
            # Iniciar animação em uma thread separada
            
            animation_thread = threading.Thread(target=animation_loop)
            animation_thread.daemon = True  # A thread será encerrada quando o programa principal terminar
            animation_thread.start()
            
            # Executar o LLM com os documentos recuperados
            results = generation_pipeline.run(
                {
                    "prompt_builder": {
                        "query": query, 
                        "documents": retrieval_results['retriever']['documents']
                    }
                }
            )
            
            # Parar a animação
            stop_animation.set()
            if animation_thread and animation_thread.is_alive():
                animation_thread.join(timeout=1.0)  # Aguardar a thread finalizar
            
        except KeyboardInterrupt:
            # Garantir que a animação pare se o usuário interromper
            if 'stop_animation' in locals() and animation_thread and animation_thread.is_alive():
                stop_animation.set()
                animation_thread.join(timeout=1.0)
            print("\n❌ Operação cancelada pelo usuário.")
            continue
        except Exception as e:
            # Garantir que a animação pare se houver um erro
            if 'stop_animation' in locals() and animation_thread and animation_thread.is_alive():
                stop_animation.set()
                animation_thread.join(timeout=1.0)
            raise  # Re-lança a exceção para ser tratada pelo bloco except externo

        processing_time = time.time() - start_time
        print(f"\n✅ Resposta gerada em {processing_time:.2f}s")

        # Imprimir a resposta gerada pelo LLM Ollama
        if "llm" in results and "replies" in results["llm"] and results["llm"]["replies"]:
            resposta_gerada = results["llm"]["replies"][0]
            print("\n📝 Resposta:")
            print("-" * 80)
            print(resposta_gerada.strip())
            print("-" * 80)
        else:
            print("\n❌ O modelo não gerou uma resposta.")
            print(f"[DEBUG] Conteúdo do resultado: {results}")

    except requests.exceptions.Timeout:
        print(f"\n⏱️ Timeout na requisição ao Ollama após {OLLAMA_TIMEOUT}s.")
        print("Tente aumentar o valor de OLLAMA_TIMEOUT ou verificar o servidor.")
    except requests.exceptions.ConnectionError:
        print(f"\n❌ Erro de conexão com o servidor Ollama em {OLLAMA_BASE_URL}")
        print("Verifique se o servidor está rodando com 'ollama serve'")
    except Exception as e:
        print(f"\n❌ Ocorreu um erro: {e}")
        import traceback
        traceback.print_exc()

print("\nEncerrando.")