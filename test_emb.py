from emb import EmbeddingGemmaONNXEmbedder
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import PDFReader
import time

# universal_cleaner_min.py
import re, unicodedata
from collections import Counter

ZW = ''.join(map(chr,[0x200B,0x200C,0x200D,0x2060,0xFEFF]))
ZW_RE   = re.compile(f"[{re.escape(ZW)}]")
ELL_RE  = re.compile(r"[.\·∙•…_]{4,}")
TAB_H   = re.compile(r"(\||\t)")
TAB_SP  = re.compile(r"\s{2,}.*\s{2,}")
ALNUM2  = re.compile(r"[A-Za-zÀ-ỹ0-9]{2,}")
END_S   = re.compile(r"[.!?;:»”’)\]]$")
BUL     = re.compile(r"^\s*([\-–—•·]|\d+[\.\)]|\([a-zA-Z0-9]\))\s+")
URL     = re.compile(r"https?://|www\.", re.I)
MAIL    = re.compile(r"[\w\.-]+@[\w\.-]+")
PHONE   = re.compile(r"\+?\d[\d\-\s]{6,}\d")

def _is_table(ln:str)->bool:
    return bool(TAB_H.search(ln) or (TAB_SP.search(ln) and re.search(r"[A-Za-zÀ-ỹ]",ln) and re.search(r"\d",ln)))

def clean(text:str)->str:
    s = unicodedata.normalize("NFKC", text.replace("\r","").replace("\u00A0"," "))
    s = ZW_RE.sub("", s).translate(str.maketrans({"“":"\"","”":"\"","„":"\"","’":"'","‘":"'","—":"-","–":"-","−":"-"}))
    s = ELL_RE.sub(" ", s)
    lines = s.split("\n")

    # drop nhiễu: dòng ít chữ/số, toàn kí tự chấm/ký hiệu; hoặc chuỗi số rời rạc dài
    kept=[]
    for ln in lines:
        t = ln.strip()
        if len(t)>=6 and not(URL.search(t) or MAIL.search(t) or PHONE.search(t)):
            if not ALNUM2.search(t):
                p = sum(1 for ch in t if unicodedata.category(ch).startswith("P"))
                if p/len(t) >= 0.60: 
                    continue
            toks = t.split()
            if len(toks)>=6 and sum(tok.isdigit() for tok in toks)>=len(toks)-1 and sum(any(c.isalpha() for c in tok) for tok in toks)<=1:
                continue
        kept.append(ln)

    # khử lặp header/footer xuất hiện nhiều lần
    freq = Counter([ln.strip() for ln in kept if len(ln.strip())>=20])
    kept = [ln for ln in kept if not(len(ln.strip())>=20 and freq[ln.strip()]>=3)]

    # nối soft-wrap (giữ bảng)
    out, buf = [], []
    def flush():
        nonlocal buf
        if buf: out.append(" ".join(buf)); buf=[]
    for i,ln in enumerate(kept):
        if not ln.strip(): flush(); out.append(""); continue
        if _is_table(ln): flush(); out.append(ln); continue
        buf.append(ln.strip())
        nxt = kept[i+1].strip() if i+1<len(kept) else ""
        if END_S.search(ln.strip()) or BUL.search(nxt): flush()
    flush()

    # co khoảng trắng (không đụng dòng bảng), giới hạn 2 dòng trống liên tiếp
    out = [re.sub(r"[ \t]+"," ", ln) if not _is_table(ln) else ln.replace("\t","    ") for ln in out]
    final, blanks = [], 0
    for ln in out:
        if ln.strip()=="":
            blanks+=1
            if blanks<=2: final.append("")
        else:
            blanks=0; final.append(ln)
    return "\n".join(final).strip()

def clean_nodes_inplace(nodes):
    for nd in nodes:
        nd.text = clean(getattr(nd,"text",""))
        md = getattr(nd,"metadata",{})
        md["_cleaned_universal"]=True
        nd.metadata = md
    return nodes


# ===== Khởi tạo embedder =====
embedder = EmbeddingGemmaONNXEmbedder(
    model_dir="embeddinggemma-300m-ONNX",
    quantized=True,
    embed_batch_size=64,
    max_retries=5,
    timeout=30.0
)

file_extractor = {".docx": PDFReader()}

# data_dir="D:/project/project_vpbank/orion-retriever/resources/docs/1"
data_dir="D:/project/rerank_ms-marco-MiniLM-L12-v2/docs"

documents = SimpleDirectoryReader(
    input_dir=str(data_dir),
    file_extractor=file_extractor,
    recursive=True,
).load_data()

print("AAAAAAA", documents)

splitter = SemanticSplitterNodeParser(
    buffer_size=1,
    breakpoint_percentile_threshold=95,
    embed_model=embedder
)

# t1 = time.time()
nodes = splitter.get_nodes_from_documents(documents)
# In thử 3 node đầu
for i, nd in enumerate(nodes[:3], 1):
    meta = getattr(nd,"metadata",{})
    print(f"\n=== Node {i} | file={meta.get('file_name')} | page={meta.get('page_label')} ===")
    print(nd.text[:800])
