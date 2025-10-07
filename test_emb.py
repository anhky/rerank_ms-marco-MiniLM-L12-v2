from emb import EmbeddingGemmaONNXEmbedder
from llama_index.core.node_parser import SemanticSplitterNodeParser

# ===== Khởi tạo embedder =====
embed_model = EmbeddingGemmaONNXEmbedder(model_dir="embeddinggemma-300m-ONNX", quantized=True)

# ===== Semantic Splitter =====
splitter = SemanticSplitterNodeParser(
    buffer_size=1,
    breakpoint_percentile_threshold=95,
    embed_model=embed_model,
)

# ===== Thử nghiệm chunk =====
from llama_index.core import Document

text = """
Hành tinh Sao Hỏa thường được gọi là Hành tinh Đỏ do bề mặt của nó có màu đỏ đặc trưng,
nguyên nhân là do sự hiện diện của oxit sắt. Đây là hành tinh thứ tư tính từ Mặt Trời
và có hai vệ tinh tự nhiên là Phobos và Deimos. Nhiều tàu vũ trụ đã được gửi đến Sao Hỏa
để nghiên cứu cấu trúc địa chất, khí quyển và khả năng tồn tại sự sống trong quá khứ.
Các sứ mệnh của NASA như Curiosity và Perseverance đã mang lại nhiều phát hiện quan trọng.
"""
doc = Document(text=text)

nodes = splitter.get_nodes_from_documents([doc])
print(f"✅ Tạo {len(nodes)} chunks:")
for n in nodes:
    print(f"- {len(n.text)} chars | preview: {n.text[:60]}...")
