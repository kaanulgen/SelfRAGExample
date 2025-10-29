import os
import yaml
import logging
from typing import List
from typing_extensions import TypedDict
from pprint import pprint
from dotenv import load_dotenv

# LangChain imports (0.2.7 versiyonu için)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from pydantic import BaseModel, Field
from langgraph.graph import END, StateGraph
from langchain_community.document_loaders import ArxivLoader

import ssl, certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

# =============================================================================
# 0. CONFIGURATION LOADER
# =============================================================================

class Config:
    """Yapılandırma yöneticisi"""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # .env dosyasını yükle
        load_dotenv()

        # Logging ayarla
        self._setup_logging()

    def _setup_logging(self):
        """Logging yapılandırması"""
        log_config = self.config.get('logging', {})
        logging.basicConfig(
            level=getattr(logging, log_config.get('level', 'INFO')),
            format=log_config.get('format', '%(asctime)s - %(levelname)s - %(message)s')
        )
        self.logger = logging.getLogger(__name__)

    def get_llm_provider(self):
        """Seçili LLM provider'ı döndür"""
        return self.config.get('llm_provider', 'openai')

    def get_llm_config(self):
        """LLM yapılandırmasını döndür"""
        provider = self.get_llm_provider()
        return self.config.get(provider, {})

    def get_api_key(self):
        """API anahtarını .env'den al"""
        llm_config = self.get_llm_config()
        api_key_env = llm_config.get('api_key_env')
        api_key = os.getenv(api_key_env)

        if not api_key:
            raise ValueError(f"{api_key_env} bulunamadı. Lütfen .env dosyasını kontrol edin.")

        return api_key

    def get(self, key, default=None):
        """Yapılandırma değeri al"""
        return self.config.get(key, default)


# =============================================================================
# 1. DATA MODELS
# =============================================================================

class GraphState(TypedDict):
    """
    Graf durumunu temsil eder.

    Attributes:
        question: Kullanıcı sorusu
        generation: LLM tarafından üretilen cevap
        documents: Alınan dökümanlar listesi
    """
    question: str
    generation: str
    documents: List[str]


class GradeDocuments(BaseModel):
    """Alınan dökümanların ilgililik kontrolü için ikili skor."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


class GradeHallucinations(BaseModel):
    """Üretilen cevapta halüsinasyon olup olmadığının kontrolü için ikili skor."""
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


class GradeAnswer(BaseModel):
    """Cevabın soruyu ele alıp almadığını değerlendirmek için ikili skor."""
    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


# =============================================================================
# 2. LLM FACTORY
# =============================================================================

class LLMFactory:
    """LLM ve Embedding modellerini oluşturan factory sınıfı"""

    @staticmethod
    def create_llm(config: Config):
        """Yapılandırmaya göre LLM oluştur"""
        provider = config.get_llm_provider()
        llm_config = config.get_llm_config()
        api_key = config.get_api_key()

        if provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=llm_config['model'],
                temperature=llm_config['temperature'],
                openai_api_key=api_key
            )

        elif provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=llm_config['model'],
                temperature=llm_config['temperature'],
                google_api_key=api_key,
                convert_system_message_to_human=True
            )

        else:
            raise ValueError(f"Desteklenmeyen provider: {provider}")

    @staticmethod
    def create_embeddings(config: Config):
        """Yapılandırmaya göre embedding modeli oluştur"""
        provider = config.get_llm_provider()
        llm_config = config.get_llm_config()
        api_key = config.get_api_key()

        if provider == "openai":
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(
                model=llm_config.get('embedding_model', 'text-embedding-3-small'),
                openai_api_key=api_key
            )

        elif provider == "gemini":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            return GoogleGenerativeAIEmbeddings(
                model=llm_config.get('embedding_model', 'models/embedding-001'),
                google_api_key=api_key
            )

        else:
            raise ValueError(f"Desteklenmeyen provider: {provider}")


# =============================================================================
# 3. VECTOR STORE SETUP
# =============================================================================

def setup_vectorstore(config: Config):
    """Arxiv makalelerinden vektör deposu oluştur - Metadata temizlemeli"""
    from langchain_community.vectorstores.utils import filter_complex_metadata

    logger = logging.getLogger(__name__)
    logger.info("Vektör deposu oluşturuluyor...")

    provider = config.get_llm_provider()
    logger.info(f" Provider: {provider}")

    # Data sources
    data_sources = config.get('data_sources', {})
    arxiv_ids = data_sources.get('arxiv_ids', [])

    if not arxiv_ids:
        raise ValueError("Config'de arxiv_ids boş!")

    logger.info(f" {len(arxiv_ids)} arxiv makalesi yükleniyor...")

    # Arxiv dökümanlarını yükle
    docs = []
    stats = {
        'successful': [],
        'failed': [],
        'too_short': []
    }

    for paper_id in arxiv_ids:
        try:
            loader = ArxivLoader(
                query=paper_id,
                load_max_docs=1,
                load_all_available_meta=True,
                doc_content_chars_max=None
            )
            loaded_docs = loader.load()

            if not loaded_docs:
                logger.warning(f"    {paper_id} - Boş döküman")
                stats['failed'].append(paper_id)
                continue

            doc = loaded_docs[0]
            char_count = len(doc.page_content)
            word_count = len(doc.page_content.split())

            if char_count < 6000:
                logger.warning(f"    {paper_id} - Çok kısa: {char_count:,} karakter")
                stats['too_short'].append((paper_id, char_count))
            else:
                logger.info(f"  {paper_id} - {char_count:,} karakter, {word_count:,} kelime")
                stats['successful'].append((paper_id, char_count))
                docs.append(doc)

        except Exception as e:
            logger.error(f"  {paper_id} - Hata: {str(e)[:100]}")
            stats['failed'].append(paper_id)

    # İstatistikleri raporla
    logger.info("\n YÜKLEME İSTATİSTİKLERİ:")
    logger.info(f"   Başarılı: {len(stats['successful'])}")
    logger.info(f"   Çok kısa: {len(stats['too_short'])}")
    logger.info(f"   Başarısız: {len(stats['failed'])}")

    if stats['successful']:
        total_chars = sum(c for _, c in stats['successful'])
        avg_chars = total_chars // len(stats['successful'])
        logger.info(f"   Toplam veri: {total_chars:,} karakter")
        logger.info(f"   Ortalama: {avg_chars:,} karakter/makale")

    if not docs:
        raise ValueError(" Hiçbir döküman başarıyla yüklenemedi!")

    # Metadata temizle - None değerleri kaldır
    logger.info(" Metadata temizleniyor...")
    docs = filter_complex_metadata(docs)

    # Text splitting - DÜZELTİLMİŞ
    vector_config = config.get('vector_store', {})
    chunk_size = vector_config.get('chunk_size', 2000)
    chunk_overlap = vector_config.get('chunk_overlap', 100)

    logger.info(f"  Chunk ayarları: size={chunk_size}, overlap={chunk_overlap}")

    # Karakter bazlı splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    doc_splits = text_splitter.split_documents(docs)

    # Chunk analizi
    total_chars = sum(len(doc.page_content) for doc in docs)
    expected_chunks = total_chars / chunk_size

    logger.info(f"  {len(doc_splits)} chunk oluşturuldu (beklenen: ~{int(expected_chunks)})")

    if len(doc_splits) < expected_chunks * 0.5:
        logger.warning("  Beklenen chunk sayısından az! Kontrol edin.")

    # Embedding modeli
    embeddings = LLMFactory.create_embeddings(config)
    test_emb = embeddings.embed_query("test")
    logger.info(f" Embedding boyutu: {len(test_emb)}")

    # Provider'a özel koleksiyon
    base_collection_name = vector_config.get('collection_name', 'rag-chroma')
    collection_name = f"{base_collection_name}-{provider}"

    persist_dir = vector_config.get('persist_directory', './chroma_db')
    os.makedirs(persist_dir, exist_ok=True)

    logger.info(f" Koleksiyon: {collection_name}")

    # Vectorstore oluştur
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir
    )

    # Retriever
    retrieval_config = config.get('retrieval', {})
    top_k = retrieval_config.get('top_k', 6)
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    logger.info(f" Vectorstore hazır: {len(doc_splits)} chunk, top_k={top_k}\n")
    return retriever


# =============================================================================
# 4. GRADERS AND CHAINS
# =============================================================================

def create_retrieval_grader(config: Config):
    """Belge ilgililiğini değerlendiren grader oluştur"""
    llm = LLMFactory.create_llm(config)

    prompts = config.get('prompts', {})
    system_prompt = prompts.get('retrieval_grader',
                                "You are a grader assessing relevance of a retrieved document to a user question. If the document contains keywords or semantic content related to the question, return 'yes'. Otherwise return 'no'. Return only a single word: 'yes' or 'no'.")

    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ])

    def parse_grade_response(response):
        response_text = response.content.strip().lower()
        return "yes" if "yes" in response_text else "no"

    return grade_prompt | llm | parse_grade_response

def create_rag_chain(config: Config):
    """RAG zinciri oluştur"""
    prompt = hub.pull("rlm/rag-prompt")
    llm = LLMFactory.create_llm(config)
    return prompt | llm | StrOutputParser()

def create_hallucination_grader(config: Config):
    """Halüsinasyon kontrolü için grader oluştur"""
    llm = LLMFactory.create_llm(config)

    prompts = config.get('prompts', {})
    system_prompt = prompts.get('hallucination_grader',
                                "You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. If the generation is supported by the facts, return 'yes'. Otherwise return 'no'. Return only a single word: 'yes' or 'no'.")

    hallucination_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ])

    def parse_grade_response(response):
        response_text = response.content.strip().lower()
        return "yes" if "yes" in response_text else "no"

    return hallucination_prompt | llm | parse_grade_response


def create_answer_grader(config: Config):
    """Cevap kalitesini değerlendiren grader oluştur"""
    llm = LLMFactory.create_llm(config)

    prompts = config.get('prompts', {})
    system_prompt = prompts.get('answer_grader',
                                "You are a grader assessing whether an answer addresses / resolves a question. If the answer properly addresses the question, return 'yes'. Otherwise return 'no'. Return only a single word: 'yes' or 'no'.")

    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ])

    def parse_grade_response(response):
        response_text = response.content.strip().lower()
        return "yes" if "yes" in response_text else "no"

    return answer_prompt | llm | parse_grade_response


def create_question_rewriter(config: Config):
    """Soru yeniden yazma zinciri oluştur"""
    llm = LLMFactory.create_llm(config)

    prompts = config.get('prompts', {})
    system_prompt = prompts.get('question_rewriter',
                                "You are a question re-writer that converts an input question to a better version that is optimized for vectorstore retrieval.")

    re_write_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
    ])

    return re_write_prompt | llm | StrOutputParser()

# =============================================================================
# 5. GRAPH NODES
# =============================================================================

class SelfRAGNodes:
    """Self-RAG graf düğmlerini içeren sınıf"""

    def __init__(self, retriever, retrieval_grader, rag_chain,
                 hallucination_grader, answer_grader, question_rewriter):
        self.retriever = retriever
        self.retrieval_grader = retrieval_grader
        self.rag_chain = rag_chain
        self.hallucination_grader = hallucination_grader
        self.answer_grader = answer_grader
        self.question_rewriter = question_rewriter
        self.logger = logging.getLogger(__name__)

    def retrieve(self, state):
        """Dökümanları al"""
        self.logger.info("---RETRIEVE---")
        question = state["question"]
        documents = self.retriever.invoke(question)
        return {"documents": documents, "question": question}

    def generate(self, state):
        """Cevap üret"""
        self.logger.info("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        generation = self.rag_chain.invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}

    def grade_documents(self, state):
        """Alınan dökümanların soruyla ilgililiğini belirle"""
        self.logger.info("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        # Her dökümanı puanla
        filtered_docs = []
        for d in documents:

            grade_str = self.retrieval_grader.invoke({
                "question": question,
                "document": d.page_content
            })

            grade = grade_str
            if grade == "yes":
                self.logger.info("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                self.logger.info("---GRADE: DOCUMENT NOT RELEVANT---")

        return {"documents": filtered_docs, "question": question}

    def transform_query(self, state):
        """Soruyu daha iyi hale getir"""
        self.logger.info("---TRANSFORM QUERY---")
        question = state["question"]
        documents = state["documents"]
        better_question = self.question_rewriter.invoke({"question": question})
        self.logger.info(f"Transformed: '{question}' → '{better_question}'")
        return {"documents": documents, "question": better_question}

    def decide_to_generate(self, state):
        """Cevap üretilecek mi yoksa soru yeniden mi yazılacak karar ver"""
        self.logger.info("---ASSESS GRADED DOCUMENTS---")
        filtered_documents = state["documents"]

        if not filtered_documents:
            self.logger.info("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
            return "transform_query"
        else:
            self.logger.info("---DECISION: GENERATE---")
            return "generate"

    def grade_generation_v_documents_and_question(self, state):
        """Üretilen cevabın dökümanlara dayalı olup olmadığını ve soruyu cevapladığını kontrol et"""
        self.logger.info("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        # Halüsinasyon kontrolü
        score_str = self.hallucination_grader.invoke({
            "documents": documents,
            "generation": generation
        })

        grade = score_str

        if grade == "yes":
            self.logger.info("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # Soru-cevap kontrolü
            self.logger.info("---GRADE GENERATION vs QUESTION---")

            score_str_2 = self.answer_grader.invoke({
                "question": question,
                "generation": generation
            })

            grade = score_str_2
            if grade == "yes":
                self.logger.info("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                self.logger.info("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            self.logger.info("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"


# =============================================================================
# 6. GRAPH CONSTRUCTION
# =============================================================================

def build_graph(nodes):
    """Self-RAG grafını oluştur"""
    workflow = StateGraph(GraphState)

    # Düğümleri ekle
    workflow.add_node("retrieve", nodes.retrieve)
    workflow.add_node("grade_documents", nodes.grade_documents)
    workflow.add_node("generate", nodes.generate)
    workflow.add_node("transform_query", nodes.transform_query)

    # Başlangıç noktasını tanımla
    workflow.set_entry_point("retrieve")

    # Akışı tanımla
    workflow.add_edge("retrieve", "grade_documents")

    workflow.add_conditional_edges(
        "grade_documents",
        nodes.decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )

    workflow.add_edge("transform_query", "retrieve")

    workflow.add_conditional_edges(
        "generate",
        nodes.grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "transform_query",
        },
    )

    # Grafı derle
    return workflow.compile()


# =============================================================================
# 7. MAIN EXECUTION
# =============================================================================

def main(config_path: str = "config.yaml"):
    """Ana çalıştırma fonksiyonu"""

    # Yapılandırmayı yükle
    config = Config(config_path)
    logger = logging.getLogger(__name__)

    provider = config.get_llm_provider()
    logger.info(f"LLM Provider: {provider}")
    logger.info(f"Model: {config.get_llm_config()['model']}")

    # Vektör deposunu oluştur
    retriever = setup_vectorstore(config)

    # Grader'ları ve zincirleri oluştur
    logger.info("Grader'lar ve zincirler oluşturuluyor...")
    retrieval_grader = create_retrieval_grader(config)
    rag_chain = create_rag_chain(config)
    hallucination_grader = create_hallucination_grader(config)
    answer_grader = create_answer_grader(config)
    question_rewriter = create_question_rewriter(config)

    # Node'ları oluştur
    nodes = SelfRAGNodes(
        retriever=retriever,
        retrieval_grader=retrieval_grader,
        rag_chain=rag_chain,
        hallucination_grader=hallucination_grader,
        answer_grader=answer_grader,
        question_rewriter=question_rewriter
    )

    # Grafı oluştur
    logger.info("Graf oluşturuluyor...")
    app = build_graph(nodes)

    # Test sorularını al
    test_questions = config.get('test_questions', [
        "Explain how the different types of agent memory work?"
    ])

    # Her soru için çalıştır
    for question in test_questions:
        print("\n" + "=" * 80)
        print(f"QUESTION: {question}")
        print("=" * 80 + "\n")

        inputs = {"question": question}
        for output in app.stream(inputs):
            for key, value in output.items():
                print(f"Node '{key}':")
            print("\n---\n")

        # Final cevabı yazdır
        print("\n" + "=" * 80)
        print("FINAL GENERATION:")
        print("=" * 80)
        pprint(value["generation"])
        print("\n")


if __name__ == "__main__":
    import sys

    # Komut satırından config dosyası yolu alabilir
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"

    try:
        main(config_path)
    except Exception as e:
        logging.error(f"Hata oluştu: {str(e)}")
        raise