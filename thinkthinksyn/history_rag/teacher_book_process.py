from __future__ import annotations

import os
import re
from pathlib import Path
import sys
from pathlib import Path
import asyncio
from typing import List, Dict

from docx import Document
from pydantic import BaseModel, Field, TypeAdapter # type: ignore
import orjson
from aiohttp import ClientResponseError  # 用于重试时捕获 HTTP 错误

from docx import Document
from docx.document import Document as _Document
from docx.table import Table as _Table, _Cell
from docx.text.paragraph import Paragraph as _Paragraph
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P

# 工程根路径（按照你原来的写法）
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
from thinkthinksyn import ThinkThinkSyn


# ========= 1. Pydantic 模型 =========

class QAItem(BaseModel):
    label: str | None = None          # 题号或小标题，例如“课题一”“问题一”“2”
    question: str | None = None       # 题干，实在没有也可以留空
    answer: str | None = None         # 参考答案，找不到就留空


class TeacherChunk(BaseModel):
    # page 保存完整页码字符串，例如“必修 第一册 P.76”
    page: str | None = None           # 完整页码字符串
    page_num: int | None = None       # 纯数字页码 76，方便后面对齐学生版
    qas: List[QAItem] = Field(default_factory=list)    # 本页抽取的题目和答案
    title: str | None = None          # 本页大标题，可选
    content: str | None = None        # 本页内容的简要概括，可选

# t = TypeAdapter(QAItem | TeacherChunk)  # 预先生成 JSON schema，避免每次调用时重复生成
# t.json_schema()
# TypeAdapter

# ========= 2. 按“必修 第一册 P.xx”拆分 docx =========
# 同时保留“必修 第一册 P.76”这个完整字符串
# 注意：教师用书里可能是“册”也可能是“冊”，这里都兼容.大小写的p也都支持

PAGE_PATTERN = re.compile(
    r"(必修\s*第[一二三四五六七八九十]+\s*[\u518c\u518a]\s*[pP]\.(\d+))"
)
# group(1) = "必修 第一册/冊 P.76"
# group(2) = "76"


def _iter_block_items(parent):
    """
    按文档真实顺序遍历：段落 + 表格
    来自 python-docx 官方 FAQ 的写法做了点精简
    """
    if isinstance(parent, _Document):
        parent_elm = parent.element.body
    elif isinstance(parent, _Cell):
        parent_elm = parent._tc
    else:
        raise ValueError("未知容器类型: %s" % type(parent))

    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield _Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield _Table(child, parent)


def split_docx_by_page(docx_path: str) -> Dict[int, Dict[str, str]]:
    """
    把教师用书 docx 拆成：
    {
        页码数字: {
            "label": 完整标签字符串,
            "text": 该页全部文字（包括表格内容）
        }
    }
    """
    doc = Document(docx_path)
    pages: Dict[int, Dict[str, str]] = {}

    current_page_num: int | None = None
    current_page_label: str | None = None
    buffer: List[str] = []

    def flush_page():
        nonlocal buffer, current_page_num, current_page_label
        if current_page_num is None:
            return
        page_text = "\n".join(buffer).strip()
        if current_page_num in pages:
            if page_text:
                pages[current_page_num]["text"] += "\n" + page_text
        else:
            pages[current_page_num] = {
                "label": current_page_label or f"P.{current_page_num}",
                "text": page_text,
            }
        buffer = []

    def process_text(text: str):
        nonlocal current_page_num, current_page_label, buffer
        text = text.strip()
        if not text:
            return
        m = PAGE_PATTERN.search(text)
        if m:
            # 碰到新的页码，先收尾上一页
            new_label = m.group(1).strip()
            new_num = int(m.group(2))
            flush_page()
            current_page_num = new_num
            current_page_label = new_label
        else:
            if current_page_num is not None:
                buffer.append(text)

    # 1) 按顺序遍历文档中的段落和表格
    for block in _iter_block_items(doc):
        if isinstance(block, _Paragraph):
            process_text(block.text)
        elif isinstance(block, _Table):
            # 表格中每个单元格的文本也按顺序处理
            for row in block.rows:
                for cell in row.cells:
                    # cell.text 可能包含多行，逐行处理更保险
                    for line in cell.text.splitlines():
                        process_text(line)

    # 2) 文档末尾收尾：即使最后一页只有表格/页眉也要记录
    if current_page_num is not None:
        flush_page()

    return pages


# ========= 3. ThinkThinkSyn 客户端 + 带重试的 completion =========

tts = ThinkThinkSyn(
    apikey=os.getenv(
        "TTS_APIKEY",
        "bq8m4hvTBbxRqJ2sA3B8ISEVJclgmsCtJaa4YTlYWYaK4JHIoP8BROjVnQNcs47B",
    )
)

MAX_CHARS = 4000  # 每页最多送这么多字符，避免 prompt 过长拖超时


async def completion_with_retry(*args, max_retries: int = 3, **kwargs):
    """
    对 tts.completion 做一层简单重试封装：
    - 如果遇到 429/5xx/504，则等待一段时间后重试
    - 超过最大重试次数后再抛出异常
    """
    for attempt in range(max_retries):
        try:
            return await tts.completion(*args, **kwargs)
        except ClientResponseError as e:
            if e.status in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                wait = 2 ** attempt  # 1, 2, 4 秒退避
                print(f"请求失败（status={e.status}），第 {attempt+1} 次重试，等待 {wait} 秒……")
                await asyncio.sleep(wait)
                continue
            raise


async def extract_from_teacher(page_num: int, page_label: str, teacher_text: str) -> TeacherChunk:
    schema = TeacherChunk.model_json_schema()

    # 限制单页输入长度，避免文本太长导致超时
    short_text = teacher_text[:MAX_CHARS]

    prompt = (
        f"下面是一段历史教师用书中对应“{page_label}”这一页的内容，"
        "请从中提取本页所有有意义的练习题及其参考答案，并按照给定的 JSON schema 返回 TeacherChunk 结构。\n"
        "【说明】\n"
        "1. 练习题通常出现在“练习要旨”“问题拆解”“课前准备”等小节中。\n"
        "2. 题目的常见形式包括：\n"
        "   - 以“问题一：”“问题二：”开头的句子；\n"
        "   - 以数字加句点开头的句子，例如“2. 细阅资料，找出有关措施，然后想想……”。\n"
        "3. “课前准备答案”下面的每一行，例如“问题一：皇帝”“问题二：推行郡县制度……”等，"
        "都是前面题目的参考答案。\n"
        "4. 请为每一道题构造一个 QAItem：\n"
        "   - label：写“问题一”“问题二”或“2”等能标识题目的字样；\n"
        "   - question：填题干原文。若题干只出现了一部分，请把能看到的部分都写入；"
        "若确实找不到题干，可以设为空字符串；\n"
        "   - answer：填在“课前准备答案”等部分中出现的对应答案内容；\n"
        "5. 如果这一页没有任何题目或答案，请让 qas 为一个空数组。\n\n"
        f"JSON schema:\n{schema}\n\n"
        "教师用书内容如下：\n"
        f"{short_text}"
    )

    result = await completion_with_retry(
        messages=[{"role": "user", "content": prompt}],
        json_schema=schema,
        model_filter="name == Qwen/Qwen3-Omni-30B-A3B-Instruct",
    )

    chunk = TeacherChunk.model_validate_json(result["text"])  # type: ignore
    # 强制覆盖 page 和 page_num，保证和拆页时一致
    chunk.page = page_label
    chunk.page_num = page_num
    return chunk


# ========= 4. 保存 / 读取（使用 orjson，支持续传） =========

def save_teacher_chunks_to_json(teacher_chunks: Dict[int, TeacherChunk], json_path: str):
    """
    将 {页码: TeacherChunk} 保存到一个 json 文件中。
    json 结构为一个列表，每个元素是一页的内容。
    使用 orjson 进行序列化。
    """
    data = []
    # 按页码排序，保证输出稳定
    for page_num in sorted(teacher_chunks.keys()):
        chunk = teacher_chunks[page_num]
        data.append(chunk.model_dump())

    with open(json_path, "wb") as f:
        f.write(orjson.dumps(
            data,
            option=orjson.OPT_INDENT_2
        ))

def load_teacher_chunks_from_json(json_path: str) -> Dict[int, TeacherChunk]:
    """
    从 json 文件中加载 {页码: TeacherChunk} 结构。
    使用 orjson 进行反序列化。
    若文件不存在，返回空字典。
    """
    if not os.path.exists(json_path):
        return {}

    with open(json_path, "rb") as f:
        data = orjson.loads(f.read())

    chunks: Dict[int, TeacherChunk] = {}
    for item in data:
        chunk = TeacherChunk(**item)
        if chunk.page_num is not None:
            chunks[chunk.page_num] = chunk
    return chunks

# ========= 5. 整本书处理：支持断点续传 & 每页即时保存 =========

async def build_teacher_chunks(
    docx_path: str,
    json_path: str,
) -> Dict[int, TeacherChunk]:
    """
    处理整本教师用书：
    - 若 json_path 已有部分结果，则先加载，作为已完成页，直接跳过（断点续传）。
    - 每处理完一页，立即更新 teacher_chunks 并写回 json_path（即时保存）。
    """
    page_info = split_docx_by_page(docx_path)

    # 1) 尝试加载已有结果，实现续传
    teacher_chunks: Dict[int, TeacherChunk] = load_teacher_chunks_from_json(json_path)
    finished_pages = set(teacher_chunks.keys())
    print(f"已完成页数: {len(finished_pages)}，将跳过这些页。")

    # 2) 按页码顺序处理
    for page_num in sorted(page_info.keys()):
        if page_num in finished_pages:
            # print(f"页面 P.{page_num} 已存在，跳过。")
            continue

        info = page_info[page_num]
        page_label = info["label"]
        text = info["text"]
        print(f"processing teacher page {page_label} ...")

        try:
            chunk = await extract_from_teacher(
                page_num=page_num,
                page_label=page_label,
                teacher_text=text,
            )
        except Exception as e:
            # 某一页出错，不让整本书崩掉，打印错误后继续下一页
            print(f"页面 {page_label} 处理失败，跳过。错误：{e}")
            continue

        teacher_chunks[page_num] = chunk

        # 3) 每页处理完就立刻保存到 json，实现即时落盘
        save_teacher_chunks_to_json(teacher_chunks, json_path)
        print(f"页面 {page_label} 已写入 {json_path}。")

    return teacher_chunks

async def process_docx_folder(folder_path: str, output_folder: str):
    """
    遍历文件夹中的所有 .docx 文件，并处理每个文件生成对应的 JSON 文件。
    """
    folder_path = Path(folder_path)  # type: ignore
    output_folder = Path(output_folder)  # type: ignore

    if not output_folder.exists():  # type: ignore
        output_folder.mkdir(parents=True)  # type: ignore

    # 获取所有 .docx 文件
    docx_files = [file for file in folder_path.glob("*.docx")]  # type: ignore
    print(f"找到 {len(docx_files)} 个 .docx 文件，将处理这些文件...")

    for docx_file in docx_files:
        output_json = output_folder / f"{docx_file.stem}_teacher_chunks.json"  # type: ignore
        print(f"正在处理文件: {docx_file}，输出路径: {output_json}")

        try:
            # 调用处理单个 docx 文件的函数
            teacher_chunks = await build_teacher_chunks(docx_file, output_json)
            print(f"处理完成，文件已保存到: {output_json}")
        except Exception as e:
            print(f"处理文件 {docx_file} 时出错：{e}")


# ========= 6. 测试入口 =========

async def main():
    # 教师用书路径
    teacher_docx = r"D:\Code\thinkthinksyn\thinkthinksyn\history_rag\data\teacher\docx_json\Red_Core_BK2_A5_ch1.docx"
    # 保存结果的 json 路径
    output_json = r"D:\Code\thinkthinksyn\thinkthinksyn\history_rag\data\teacher\docx_json\Red_Core_BK2_A5_ch1_teacher_chunks.json"

    teacher_chunks = await build_teacher_chunks(teacher_docx, output_json)
    print(f"最终共解析页数: {len(teacher_chunks)}，已保存到: {output_json}")

    # 示例：打印 P.76 的结果
    sample_page = 76
    if sample_page in teacher_chunks:
        print(
            teacher_chunks[sample_page].model_dump_json(
                ensure_ascii=False,
                indent=2
            )
        )
    else:
        print(f"page {sample_page} not found in teacher doc.")

async def main_with_folder():
    # 文件夹路径和输出文件夹路径
    input_folder = r"D:\Code\thinkthinksyn\thinkthinksyn\history_rag\data\teacher\docx"
    output_folder = r"D:\Code\thinkthinksyn\thinkthinksyn\history_rag\data\teacher\docx_json"

    await process_docx_folder(input_folder, output_folder)

if __name__ == "__main__":
    asyncio.run(main_with_folder())
    # asyncio.run(main())
