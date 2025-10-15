# rag_pipeline.py

# ==============================================================================
# BƯỚC 0: IMPORT CÁC THƯ VIỆN CẦN THIẾT
# ==============================================================================
import os
import regex as re
import logging
from typing import List

# Import các thành phần chính từ LlamaIndex
from llama_index.core import SimpleDirectoryReader, Document, Settings
from llama_index.readers.file import PDFReader, DocxReader
from llama_index.core.node_parser import SemanticSplitterNodeParser, TokenTextSplitter
from llama_index.core.utils import get_tokenizer

# Import embedder tùy chỉnh (giả định file emb.py tồn tại)
from emb import EmbeddingGemmaONNXEmbedder

# Cấu hình logging để xem thông tin xử lý
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==============================================================================
# BƯỚC 1: ĐỊNH NGHĨA CÁC THAM SỐ CẤU HÌNH
# ==============================================================================
# --- Đường dẫn ---
DATA_DIR = os.getenv("DATA_DIR", "D:/project/rerank_ms-marco-MiniLM-L12-v2/docs")
EMB_DIR = os.getenv("EMBED_WEIGHTS", "embeddinggemma-300m-ONNX")

# --- Cấu hình Chunking ---
# Kích thước chunk mục tiêu (tính bằng token) cho splitter lớp 2.
TARGET_CHUNK_SIZE = 512
# Số token chồng lấn giữa các chunk để giữ ngữ cảnh.
CHUNK_OVERLAP = 50
# Ngưỡng phân tách của Semantic Splitter. Giá trị càng cao (gần 100),
# splitter càng "lười" và có xu hướng tạo ra các chunk lớn hơn về mặt ngữ nghĩa.
SEMANTIC_SPLITTER_BREAKPOINT_THRESHOLD = 95

# --- Cấu hình Embedding Model ---
EMBED_BATCH_SIZE = 64

# ==============================================================================
# BƯỚC 2: KHỞI TẠO CÁC THÀNH PHẦN CHÍNH
# ==============================================================================
logging.info("Khởi tạo các thành phần của pipeline...")

# --- Khởi tạo Tokenizer ---
# Lấy tokenizer một lần và tái sử dụng để tối ưu hiệu suất.
# Tokenizer này sẽ được dùng để đếm token và cho TokenTextSplitter.
tokenizer = get_tokenizer()

# --- Khởi tạo Embedding Model ---
# Đây là model dùng để chuyển đổi văn bản thành vector số.
# Nó được sử dụng bởi SemanticSplitter để tìm điểm ngắt ngữ nghĩa.
embedder = EmbeddingGemmaONNXEmbedder(
    model_dir=EMB_DIR,
    quantized=True,
    embed_batch_size=EMBED_BATCH_SIZE
)
# Cấu hình embedder này làm mặc định cho toàn bộ LlamaIndex
Settings.embed_model = embedder
logging.info(f"Đã khởi tạo Embedding Model từ: {EMB_DIR}")

# --- Khởi tạo các Node Parser (Splitters) ---
# LỚP 1: SemanticSplitterNodeParser
# Ưu tiên chia văn bản dựa trên sự thay đổi về ngữ nghĩa.
# Chúng ta không đặt chunk_size_limit ở đây để nó tự do nhóm các câu.
semantic_splitter = SemanticSplitterNodeParser(
    buffer_size=1,
    breakpoint_percentile_threshold=SEMANTIC_SPLITTER_BREAKPOINT_THRESHOLD,
    embed_model=embedder,
)

# LỚP 2: TokenTextSplitter
# Dùng làm "lưới an toàn" để chia nhỏ các chunk quá lớn từ lớp 1,
# đảm bảo không có chunk nào vượt quá giới hạn token.
token_splitter = TokenTextSplitter(
    chunk_size=TARGET_CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separator=" ",
    tokenizer=tokenizer,
)
logging.info("Đã khởi tạo xong các splitter.")


# ==============================================================================
# BƯỚC 3: ĐỊNH NGHĨA CÁC HÀM XỬ LÝ
# ==============================================================================

def clean_document_text(docs: List[Document]) -> List[Document]:
    """
    Hàm này nhận vào một danh sách các Document và thực hiện làm sạch trường text.
    Loại bỏ các ký tự không mong muốn và khoảng trắng thừa.
    """
    logging.info(f"Bắt đầu làm sạch văn bản cho {len(docs)} tài liệu...")
    
    # Định nghĩa các biểu thức chính quy (Regex) để làm sạch
    zw_re = re.compile(r"[\u200B\u200C\u200D\u2060\uFEFF\u00AD]")  # Ký tự không có độ rộng
    nbsp_re = re.compile(r"\u00A0")  # Khoảng trắng không ngắt dòng
    fill_line_re = re.compile(r"^(\s*[._·∙•\-–—=~]{3,}\s*)+$", re.MULTILINE) # Dòng kẻ trang trí
    ell_re = re.compile(r"[.\·∙•…_]{4,}")  # Dấu ba chấm dài
    blank3_re = re.compile(r"\n{3,}")  # 3+ dòng trống
    space2_re = re.compile(r" {2,}") # 2+ dấu cách

    clean_docs = []
    for d in docs:
        t = d.text or ""
        t = zw_re.sub("", t)
        t = nbsp_re.sub(" ", t)
        t = fill_line_re.sub("", t)
        t = ell_re.sub("...", t)
        t = blank3_re.sub("\n\n", t)
        t = space2_re.sub(" ", t)
        
        # Tạo lại Document với văn bản đã được làm sạch
        clean_docs.append(Document(text=t.strip(), metadata=d.metadata, id_=d.id_))
        
    logging.info("Hoàn tất làm sạch văn bản.")
    return clean_docs

# ==============================================================================
# BƯỚC 4: THỰC THI PIPELINE XỬ LÝ
# ==============================================================================

# --- Tải dữ liệu ---
logging.info(f"Bắt đầu tải tài liệu từ thư mục: {DATA_DIR}")
reader = SimpleDirectoryReader(
    input_dir=DATA_DIR,
    file_extractor={".pdf": PDFReader(), ".docx": DocxReader()},
    recursive=True,
)
documents = reader.load_data()
logging.info(f"Đã tải thành công {len(documents)} tài liệu.")

# --- Làm sạch văn bản ---
cleaned_documents = clean_document_text(documents)

# --- Chia nhỏ văn bản (Chunking 2 lớp) ---
logging.info("Bắt đầu quá trình chunking 2 lớp...")
# Lớp 1: Lấy các node dựa trên ngữ nghĩa
semantic_nodes = semantic_splitter.get_nodes_from_documents(cleaned_documents)
logging.info(f"Semantic Splitter đã tạo ra {len(semantic_nodes)} node ban đầu.")

# Lớp 2: Xử lý các node quá lớn bằng Token Splitter
final_nodes = []
for node in semantic_nodes:
    # Đếm số token của node hiện tại một cách hiệu quả
    token_count = len(tokenizer(node.get_content() or ""))
    
    if token_count > TARGET_CHUNK_SIZE:
        # Nếu node quá lớn, dùng token_splitter để chia nhỏ nó
        sub_nodes = token_splitter.get_nodes_from_documents([node])
        final_nodes.extend(sub_nodes)
    else:
        # Nếu node đã đạt yêu cầu, giữ nguyên nó
        final_nodes.append(node)
logging.info(f"Quá trình chunking hoàn tất. Tổng số node cuối cùng: {len(final_nodes)}")

# ==============================================================================
# BƯỚC 5: IN KẾT QUẢ VÀ THỐNG KÊ
# ==============================================================================
print("\n" + "="*50)
print(f"THỐNG KÊ KẾT QUẢ CHUNKING")
print("="*50)
print(f"Tổng số node cuối cùng: {len(final_nodes)}")
print(f"Kích thước chunk mục tiêu: {TARGET_CHUNK_SIZE} tokens")
print("="*50 + "\n")

warnings = 0
for i, node in enumerate(final_nodes, 1):
    token_count = len(tokenizer(node.get_content() or ""))
    
    print(f"--- NODE {i:03d} | Tokens: {token_count} ---")
    print(node.get_content())
    print("-" * (25 + len(str(i)) + len(str(token_count))))
    
    # Kiểm tra lại lần cuối xem có node nào vẫn vượt ngưỡng không
    if token_count > TARGET_CHUNK_SIZE:
        print(f"    ^ CẢNH BÁO: Node này vẫn lớn hơn {TARGET_CHUNK_SIZE} token!")
        warnings += 1

print(f"\nHoàn tất. Tìm thấy {warnings} cảnh báo.")