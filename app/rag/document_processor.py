import os
import shutil
from typing import Any, Dict, List, Tuple

import requests
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings

# ==========================================
# RAG 核心服务 (文档处理与向量检索)
# ==========================================
#
# RAG = Retrieval-Augmented Generation，中文通常叫“检索增强生成”。
# 一个典型 RAG 流程可以拆成两条链路：
#
# 1. 入库链路：用户上传文档 -> 读取文档 -> 切成小块 -> 生成向量 -> 存入向量库
#    对应本文件里的 process_and_store_document()。
#
# 2. 检索链路：用户提问 -> 把问题也生成向量 -> 到向量库找相似片段 -> 拼成上下文
#    对应本文件里的 retrieve_relevant_context()。
#
# 大模型本身不直接“记住”上传的文件内容。它回答时需要我们先把相关文档片段查出来，
# 再把这些片段作为 context 传给大模型，让大模型基于这些材料回答。


# 本地向量库保存目录。
#
# FAISS 会在这个目录下保存索引文件和文档映射信息。保存之后，下次程序重启仍然可以
# 重新加载已有的向量库，不需要每次都重新处理所有文档。
VECTOR_STORE_DIR = "app/rag/vectorstore"


# DashScope 文本向量 API 地址。
#
# 这里用的是阿里云 DashScope 的 embedding 服务。Embedding 的作用是把文本转换成
# 一组浮点数，也就是“向量”。语义相近的文本，它们的向量在空间中的距离通常也更近。
DASHSCOPE_EMBEDDING_URL = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"


def _dashscope_batch_size(model_name: str) -> int:
    """根据不同 embedding 模型选择单次请求的批量大小。

    DashScope 不同 embedding 模型对单次请求文本数量有不同限制。
    这里把规则封装成一个函数，后续如果模型限制变化，只需要改这一个地方。

    Args:
        model_name: settings.EMBEDDING_MODEL 中配置的 embedding 模型名称。

    Returns:
        int: 每次请求 DashScope API 时最多提交多少段文本。
    """
    # text-embedding-v3 / v4 的单批文本数量限制更小，因此设置为 6。
    # 如果一次提交太多文本，API 可能会返回参数错误或超过限制。
    if model_name in {"text-embedding-v3", "text-embedding-v4"}:
        return 6

    # 其他模型使用更大的批量，减少 HTTP 请求次数，提高入库速度。
    return 25


def _dashscope_embed_texts(texts: List[str], text_type: str) -> List[List[float]]:
    """调用 DashScope API，把一组文本转换成向量。

    这个函数是本文件里真正发起 embedding 请求的地方。它既可以处理文档片段，
    也可以处理用户问题，区别由 text_type 控制：

    - text_type="document"：表示这些文本是要入库的文档内容。
    - text_type="query"：表示这段文本是用户搜索/提问的查询内容。

    很多 embedding 服务会针对 document 和 query 使用略有不同的内部优化。
    所以检索时通常建议：文档入库用 document，用户问题用 query。

    Args:
        texts: 待向量化的文本列表。列表里每个元素会得到一个向量。
        text_type: 文本类型，通常是 "document" 或 "query"。

    Returns:
        List[List[float]]: 向量列表。返回顺序与传入 texts 的顺序保持一致。

    Raises:
        RuntimeError: 当 DashScope API 返回非 200 状态码时抛出，便于上层发现问题。
    """
    # DashScope 使用 Bearer Token 鉴权。
    #
    # 注意：这里读取的是 settings.OPENAI_API_KEY。这个项目可能复用了 OpenAI 风格
    # 的配置字段名，但实际请求地址是 DashScope。学习时要区分“变量名”和“真实服务”。
    headers = {
        "Authorization": f"Bearer {settings.EMBEDDING_API_KEY}",
        "Content-Type": "application/json",
    }

    # 从配置中读取 embedding 模型名称，便于在 .env 或配置文件中切换模型。
    model_name = settings.EMBEDDING_MODEL

    # 根据模型名称决定每次请求最多发送多少条文本。
    batch_size = _dashscope_batch_size(model_name)

    # 用来累计所有批次返回的向量。
    #
    # 类型是 List[List[float]]：
    # - 外层 List：多段文本
    # - 内层 List[float]：单段文本对应的向量
    vectors: List[List[float]] = []

    # range(0, len(texts), batch_size) 会生成 0、batch_size、2*batch_size...
    # 这样可以把 texts 按 batch_size 切成多个小批次请求。
    #
    # 例如 len(texts)=13，batch_size=6，则会请求 3 次：
    # - texts[0:6]
    # - texts[6:12]
    # - texts[12:18]，实际只取到最后 1 条
    for start in range(0, len(texts), batch_size):
        # 确保每个元素都是字符串，避免上传了数字、None 等非字符串对象导致 API 报错。
        batch = [str(t) for t in texts[start : start + batch_size]]

        # DashScope embedding API 的请求体。
        #
        # model: 使用哪个 embedding 模型
        # input.texts: 本批次要向量化的文本列表
        # parameters.text_type: 告诉模型这些文本是 document 还是 query
        payload = {
            "model": model_name,
            "input": {"texts": batch},
            "parameters": {"text_type": text_type},
        }

        # 发起 HTTP POST 请求。
        #
        # timeout=120 表示最多等待 120 秒。文档较多时 embedding 请求可能比较慢，
        # 但也不能无限等待，否则接口卡住时应用会一直阻塞。
        resp = requests.post(DASHSCOPE_EMBEDDING_URL, headers=headers, json=payload, timeout=120)

        # 非 200 说明请求失败，例如 key 错误、模型不存在、请求体不合法或额度不足。
        # 直接抛异常比静默返回空向量更安全，因为空向量会让后续检索结果变得难以排查。
        if resp.status_code != 200:
            raise RuntimeError(f"DashScope embedding failed: {resp.status_code} - {resp.text}")

        # DashScope 返回 JSON。我们只关心 output.embeddings 字段。
        data = resp.json()
        embeddings = data.get("output", {}).get("embeddings", [])

        # API 返回的 embedding 中通常带有 text_index，表示它对应输入列表里的第几条文本。
        # 这里按 text_index 排序，确保返回向量顺序和 batch 输入文本顺序一致。
        #
        # 这一步很重要：向量库需要知道“哪个向量对应哪个文档块”。如果顺序错了，
        # 检索时可能查到 A 向量，却返回 B 文档内容。
        embeddings = sorted(embeddings, key=lambda x: x.get("text_index", 0))

        # 从每个 embedding 对象里取出真正的向量数组，追加到总结果中。
        vectors.extend([e.get("embedding", []) for e in embeddings])

    return vectors


class DashScopeEmbeddings(Embeddings):
    """把 DashScope embedding API 包装成 LangChain 可用的 Embeddings 类。

    LangChain 的向量库接口通常不直接接收一个普通函数，而是接收一个实现了
    Embeddings 协议的对象。这个类的作用就是适配 LangChain：

    - FAISS.from_documents(...) 会调用 embed_documents() 处理文档块。
    - FAISS.similarity_search(...) 会调用 embed_query() 处理用户查询。
    """

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """把多段文档文本转换成向量，用于写入向量库。"""
        return _dashscope_embed_texts(texts, text_type="document")

    def embed_query(self, text: str) -> List[float]:
        """把单个用户问题转换成向量，用于相似度检索。"""
        # _dashscope_embed_texts 总是返回“向量列表”。
        # 查询只有一条文本，所以取第 0 个元素作为最终查询向量。
        return _dashscope_embed_texts([text], text_type="query")[0]


def get_embeddings():
    """初始化 Embedding 模型对象。

    这里没有直接返回某个云厂商 SDK，而是返回上面自定义的 DashScopeEmbeddings。
    这样 LangChain 的 FAISS 向量库就能用统一接口调用 DashScope。
    """
    return DashScopeEmbeddings()


def process_and_store_document(file_path: str):
    """处理上传的文档，并把文档内容存入本地 FAISS 向量数据库。

    完整流程：

    1. 根据文件后缀选择加载器，把 PDF/TXT 读成 LangChain Document 对象。
    2. 把长文档切成多个小块，避免单个文本过长。
    3. 调用 embedding 服务，把每个文本块变成向量。
    4. 把“向量 + 原文片段 + 元数据”保存到 FAISS 本地向量库。

    Args:
        file_path: 文档的本地路径，例如 uploads/demo.pdf。

    Returns:
        int: 本次文档被切分出来的 chunk 数量。

    Raises:
        ValueError: 当文件格式不是 PDF 或 TXT 时抛出。
        RuntimeError: 当 embedding API 调用失败时，由 _dashscope_embed_texts 抛出。
    """
    # 1. 加载文档 (支持 PDF 和 TXT)
    #
    # LangChain 的 loader 会把原始文件转换成 Document 对象。
    # Document 通常包含：
    # - page_content: 这一页/这一段的文本内容
    # - metadata: 来源文件、页码等元数据
    if file_path.endswith(".pdf"):
        # PyPDFLoader 会按 PDF 页读取内容。一般一页会变成一个 Document。
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        # TextLoader 用于读取纯文本文件。
        # encoding="utf-8" 可以正确处理大多数中文文本文件。
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        # 这里主动限制格式，避免用户上传 Word、Excel、图片等当前 loader 不支持的文件。
        # 如果以后要支持 docx，可以增加 Docx2txtLoader 或 UnstructuredFileLoader。
        raise ValueError("不支持的文件格式，目前仅支持 PDF 和 TXT")

    # 真正执行文件读取。
    # documents 是 List[Document]，不是普通字符串列表。
    documents = loader.load()

    # 2. 文本切块 (Chunking)
    #
    # 为什么要切块？
    # - 大模型上下文窗口有限，不能无限塞整本书。
    # - 向量检索需要较细粒度的片段，否则一个很长文档只要局部相关，也会带回大量无关内容。
    # - 小块更容易精准匹配用户问题。
    #
    # 但块也不能太小，否则上下文信息不完整。chunk_size 和 chunk_overlap 就是在
    # “精确度”和“上下文完整性”之间做平衡。
    text_splitter = RecursiveCharacterTextSplitter(
        # 每个文本块大约 500 个字符。
        # 对中文来说“字符数”和“token 数”不是完全等价的，但可以作为简单起点。
        chunk_size=settings.RAG_CHUNK_SIZE,

        # 相邻文本块之间重叠一定字符。
        # 这样可以降低一句话或一个段落被切断后语义丢失的概率。
        chunk_overlap=settings.RAG_CHUNK_OVERLAP,

        # 分隔符优先级。
        #
        # RecursiveCharacterTextSplitter 会优先按更自然的边界切：
        # 1. "\n\n"：段落
        # 2. "\n"：换行
        # 3. "。" "！" "？"：中文句子结束
        # 4. "，"：中文逗号
        # 5. " "：空格
        # 6. ""：实在切不开时按字符硬切
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
    )

    # split_documents 会保留 Document 的 metadata，并把 page_content 切成多个小块。
    # chunks 仍然是 List[Document]，只是每个 Document 的内容更短。
    chunks = text_splitter.split_documents(documents)

    # 3. 向量化并存储
    #
    # embeddings 是 LangChain 认识的 Embeddings 对象。
    # 后续 FAISS 会自动调用它，把 chunks 里的 page_content 转成向量。
    embeddings = get_embeddings()

    # 检查是否已经存在向量库：
    #
    # - 存在：说明之前已经上传过文档。加载旧库后追加新 chunks。
    # - 不存在：说明这是第一次上传文档。用当前 chunks 新建向量库。
    if os.path.exists(VECTOR_STORE_DIR):
        # 从本地磁盘加载已有 FAISS 向量库。
        #
        # allow_dangerous_deserialization=True 是 LangChain 加载本地 FAISS 时常见参数。
        # 它允许反序列化 pickle 数据。安全注意：只应该加载自己程序生成、可信来源的索引文件，
        # 不要加载陌生用户提供的 FAISS 索引。
        vectorstore = FAISS.load_local(
            VECTOR_STORE_DIR,
            embeddings,
            allow_dangerous_deserialization=True,
        )

        # 把本次上传文档的新 chunks 追加到已有向量库。
        # add_documents 内部会调用 embeddings.embed_documents() 生成向量。
        vectorstore.add_documents(chunks)
    else:
        # 第一次创建向量库。
        #
        # from_documents 会做两件事：
        # 1. 调用 embeddings 把每个 chunk 转成向量。
        # 2. 建立 FAISS 索引，并保存 chunk 原文与 metadata 的映射关系。
        vectorstore = FAISS.from_documents(chunks, embeddings)

    # 保存到本地磁盘。
    # 保存后 VECTOR_STORE_DIR 目录中会出现 FAISS 索引和文档存储相关文件。
    vectorstore.save_local(VECTOR_STORE_DIR)

    # 返回 chunk 数量，通常可用于接口响应，例如“已处理 12 个文档片段”。
    return len(chunks)


def list_uploaded_documents() -> List[Dict[str, Any]]:
    """列出 uploads 目录中的文档文件。"""
    uploads_dir = "uploads"
    if not os.path.exists(uploads_dir):
        return []

    documents: List[Dict[str, Any]] = []
    for filename in sorted(os.listdir(uploads_dir)):
        if not filename.lower().endswith((".pdf", ".txt")):
            continue
        path = os.path.join(uploads_dir, filename)
        if not os.path.isfile(path):
            continue
        stat = os.stat(path)
        documents.append(
            {
                "filename": filename,
                "size_bytes": stat.st_size,
                "modified_at": int(stat.st_mtime),
            }
        )
    return documents


def clear_vector_store() -> None:
    """删除本地 FAISS 向量库目录。"""
    if os.path.exists(VECTOR_STORE_DIR):
        shutil.rmtree(VECTOR_STORE_DIR)


def delete_uploaded_document(filename: str) -> bool:
    """删除 uploads 目录中的某个文档文件。"""
    safe_filename = os.path.basename(filename)
    path = os.path.join("uploads", safe_filename)
    if not os.path.isfile(path):
        return False
    os.remove(path)
    return True


def rebuild_vector_store_from_uploads() -> int:
    """清空向量库后，重新处理 uploads 目录下所有 PDF/TXT 文件。"""
    clear_vector_store()
    total_chunks = 0
    for doc in list_uploaded_documents():
        path = os.path.join("uploads", doc["filename"])
        total_chunks += process_and_store_document(path)
    return total_chunks


def delete_document_and_rebuild(filename: str) -> bool:
    """删除指定文档后重建向量库，确保检索结果不再包含该文件。"""
    deleted = delete_uploaded_document(filename)
    if deleted:
        rebuild_vector_store_from_uploads()
    return deleted


def retrieve_relevant_context_with_citations(
    query: str, top_k: int | None = None
) -> Tuple[str, List[Dict[str, Any]]]:
    """检索相关片段，并返回用于 UI 展示的引用信息（文件名、页码、距离、摘要）。

    FAISS `similarity_search_with_score` 返回的 score 为距离型指标（数值越小通常越相似，
    具体含义以 LangChain/FAISS 配置为准）；前端仅作「可观测性」展示。
    """
    if not os.path.exists(VECTOR_STORE_DIR):
        return "", []

    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        VECTOR_STORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    pairs = vectorstore.similarity_search_with_score(query, k=top_k or settings.RAG_TOP_K)
    citations: List[Dict[str, Any]] = []
    parts: List[str] = []
    for doc, score in pairs:
        parts.append(doc.page_content)
        meta = doc.metadata or {}
        src = meta.get("source") or meta.get("file_path") or ""
        base = os.path.basename(str(src)) if src else "未知来源"
        preview = doc.page_content.strip().replace("\n", " ")
        if len(preview) > 220:
            preview = preview[:220] + "…"
        citations.append(
            {
                "source": base,
                "page": meta.get("page"),
                "distance": float(score),
                "snippet_preview": preview,
            }
        )

    context = "\n\n---\n\n".join(parts)
    return context, citations


def retrieve_relevant_context(query: str, top_k: int | None = None) -> str:
    """根据用户问题，从向量数据库中检索最相关的文档片段（仅正文，不含引用结构）。"""
    text, _ = retrieve_relevant_context_with_citations(query, top_k=top_k)
    return text
