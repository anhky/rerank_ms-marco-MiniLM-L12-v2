import numpy as np
from emb import EmbeddingGemmaONNXEmbedder  # file embedder bạn dán ở trên

# ===== 1. Khởi tạo model =====
embedder = EmbeddingGemmaONNXEmbedder(
    model_dir="embeddinggemma-300m-ONNX",
    quantized=True,
    embed_batch_size=32,
)

# ===== 2. Input dữ liệu =====
query1 = "thẻ tín dụng là gì"
docs1 = [
    "Hướng dẫn mở tài khoản thanh toán doanh nghiệp tại VPBank.",
    "Quy định về phát hành thẻ ghi nợ doanh nghiệp.",
    "Thông tin lãi suất tiền gửi có kỳ hạn cho khách hàng cá nhân.",
    "Thủ tục mở tài khoản công ty, yêu cầu hồ sơ và biểu mẫu.",
    "Trời hôm nay đẹp quá", 
    "Barack obama",
    "Thẻ tín dụng là một loại thẻ ngân hàng cho phép chủ thẻ tiêu trước và thanh toán lại cho ngân hàng sau"
]

query = "GLB 10432.2"

docs = [
    """
    Type:
    Bulletin announcement
    Category:
    Pricing/Fees
    Audience:
    Issuer
    Processor
    Region:
    Asia/Pacific (excluding Indonesia and Malaysia)
    Canada
    Latin America and the Caribbean (excluding Brazil)
    Middle East/Africa
    United States
    Brand:
    Mastercard®
    Debit Mastercard
    Maestro®
    Action indicator:
    Financial
    Published:
    19 August 2025
    Effective:
    1 January 2025
    1 August 2026

    Executive overview
    Mastercard is introducing the issuer fee for On-Behalf Name Match Service in the United States, Canada,
    Middle East/Africa, Latin America and the Caribbean (excluding Brazil), and Asia/Pacific (excluding
    Indonesia and Malaysia) regions. Customers may choose to participate in this service.

    Table 1: Effective date details
    Date Details
    1 January 2025  Pricing for On Behalf Name Match Service becomes effective for participants in the
                    Middle East/Africa, Latin America and the Caribbean (excluding Brazil), and Asia/Pacific
                    (excluding Indonesia and Malaysia) regions
    1 August 2026   Pricing for On Behalf Name Match Service becomes effective for participants in the
                    United States and Canada regions

    What Mastercard is doing
    Mastercard is introducing the fee to establish fair pricing for the value that On-Behalf Name Match Service
    delivers. Refer to the Billing and pricing information section in this announcement for billing and fee
    information.

    Background
    Mastercard is offering the On-Behalf Name Match Service to issuers as an option to support name validation.
    Mastercard will perform name matching on behalf of participating issuers reducing technical and maintenance
    efforts to support in-house name validation.

    Table 2: Version history
    Date Description of change
    19 August 2025  Updated the Effective date table and Billing and pricing information section to reflect the
                    correct dates for when pricing becomes effective per regions
    20 August 2024  Initial publication date

    AP/CAN/LAC/MEA/US 10246.2
    Introduction of On-Behalf Name Match Service Fees for Select Regions
    © 2025 Mastercard. Proprietary.
    """,

    """
    All rights reserved.
    AP/CAN/LAC/MEA/US 10246.2
    Introduction of On-Behalf Name Match Service Fees for Select Regions • 19 August 2025
    """,

    """
    Billing and pricing information
    The pricing for On Behalf Name Match Service is effective 1 January 2025 in the Middle East/Africa,
    Latin America and the Caribbean (excluding Brazil), and Asia/Pacific (excluding Indonesia and Malaysia)
    regions with the first billing occurring on 5 January 2025.

    The pricing for On Behalf Name Match Service is effective 1 August 2026 in the United States and Canada
    regions with the first billing occurring on 2 August 2026.

    Mastercard will bill an issuer for On-Behalf Name Match Service each time Mastercard successfully performs
    name matching on their behalf.

    Table 3: On-Behalf Name Match Service billing and pricing information
    Billing event number | Billing event name | Service ID | Rate (USD) | Frequency
    2AB1190 | Issuer On Behalf Name Match fee - AUTH | AB | 0.03 | Weekly
    2HR7100 | Issuer On Behalf Name Match fee - SMS  | HR | 0.03 | Weekly

    Mastercard will include these changes in the Pricing Guide following the effective date. Until that time,
    use this announcement as the source for information.

    Participation requirements
    Customers should contact their respective Account Management representatives or Global Customer Service
    for enrollment, onboarding, and testing requirements.

    Related information
    • CAN/US 10210.1 Revised Standards for Name Validation Support Requirements for Issuers in the Canada and
      U.S. Regions
    • AN 7402 Enhancing Account Status Inquiry to Enable Name Validation
    • AN 8429 Introducing On-Behalf Name Match Service and Enhancements to Name Validation Service
    • GLB 10432.2 Enhancing Send Data Fields and Name Validation Request
    • Relevant sections of the Authorization Manual, Customer Interface Specification, and Single Message System
      Programs and Services

    Questions
    Customers with questions about the information in this announcement should contact Global Customer Service
    using the contact information on the Technical Resource Center.

    © 2025 Mastercard. Proprietary.
    """,

    """
    U.S. Regions
    • AN 7402 Enhancing Account Status Inquiry to Enable Name Validation
    • AN 8429 Introducing On-Behalf Name Match Service and Enhancements to Name Validation Service
    • GLB 10432.2 Enhancing Send Data Fields and Name Validation Request
    • Relevant sections of the Authorization Manual, Customer Interface Specification, and Single Message System
    Programs and Services
    Questions
    Customers with questions about the information in this announcement should contact Global Customer Service
    using the contact information on the Technical Resource Center.
    © 2025 Mastercard. Proprietary.
    """
]

# ===== 3. Tạo embedding =====
q_emb = np.array(embedder._get_query_embedding(query))
d_emb = np.array(embedder._get_text_embeddings(docs))

# ===== 4. Tính điểm cosine similarity =====
scores = np.dot(d_emb, q_emb) / (
    np.linalg.norm(d_emb, axis=1) * np.linalg.norm(q_emb)
)

# ===== 5. In kết quả =====
for i, (doc, s) in enumerate(zip(docs, scores), 1):
    # print(f"[{i}] score={s:.4f} | {doc}")
    print(f"[{i}] score={s:.4f}")
