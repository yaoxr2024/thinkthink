import asyncio
import base64
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import fitz  # type: ignore
import orjson
from pydantic import BaseModel, Field, TypeAdapter

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
from thinkthinksyn import ThinkThinkSyn  # noqa

tts = ThinkThinkSyn(
    apikey=os.getenv(
        "TTS_APIKEY",
        "bq8m4hvTBbxRqJ2sA3B8ISEVJclgmsCtJaa4YTlYWYaK4JHIoP8BROjVnQNcs47B",
    )
)


class Paragraph(BaseModel):
    title: Optional[str] = None
    content: str
    children: List["Paragraph"] | None = None


Paragraph.model_rebuild()


class QAItem(BaseModel):
    label: Optional[str] = None
    question: Optional[str] = None
    answer: Optional[str] = None


class TeacherChunk(BaseModel):
    page: Optional[str] = None
    page_num: Optional[int] = None
    qas: List[QAItem] = Field(default_factory=list)
    title: Optional[str] = None
    content: Optional[str] = None


async def completion_with_retry(
    *,
    messages,
    json_schema,
    model_filter: str,
    max_retries: int = 3,
    base_delay: float = 2.0,
):
    """
    对 tts.completion 做一层简单重试：
      - 如果 status_code 是 429 / 5xx / 504，则等待一段时间后重试
      - 超过最大重试次数后抛出异常

    注意：这里假设 tts.completion 失败时会抛异常，异常上有 .status_code
          或 .response.status_code。
    """
    last_exc: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            return await tts.completion(
                messages=messages,
                json_schema=json_schema,
                model_filter=model_filter,
            )
        except Exception as e:  # 可以替换成更精确的异常类型
            last_exc = e

            status_code = getattr(e, "status_code", None)
            if status_code is None and hasattr(e, "response"):
                status_code = getattr(getattr(e, "response"), "status_code", None)

            retryable = False
            if status_code is not None:
                if status_code == 429:
                    retryable = True
                elif status_code == 504:
                    retryable = True
                elif 500 <= status_code < 600:
                    retryable = True

            if not retryable:
                # 非可重试错误，直接抛出
                raise

            if attempt >= max_retries:
                print(
                    f"[completion_with_retry] 已达到最大重试次数 {max_retries}，仍然失败，抛出异常..."
                )
                raise

            delay = base_delay * attempt  # 简单指数退避
            print(
                f"[completion_with_retry] 调用失败，status_code={status_code}，"
                f"第 {attempt} 次重试前等待 {delay} 秒..."
            )
            await asyncio.sleep(delay)

    if last_exc is not None:
        raise last_exc



async def ocr_one_page(
    page_num: int,
    img_bytes: bytes,
    teacher_chunks: Optional[List[TeacherChunk]],
    schema: dict,
) -> tuple[int, List[Paragraph]]:
    """
    对单页图片做 OCR，返回 (page_num, [Paragraph,...])

    teacher_chunks:
        - None：没有教师信息
        - List[TeacherChunk]：这一页所有教师信息
    """

    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    # 拼 prompt
    if teacher_chunks:
        teacher_blocks: List[str] = []
        for idx, tc in enumerate(teacher_chunks, start=1):
            qa_texts = []
            for qa in tc.qas:
                parts = []
                if qa.label:
                    parts.append(f"{qa.label}")
                if qa.question:
                    parts.append(f"题目：{qa.question}")
                if qa.answer:
                    parts.append(f"答案：{qa.answer}")
                if parts:
                    qa_texts.append("；".join(parts))

            block_title = tc.title or ""
            block_summary = tc.content or ""
            block_qas = "\n".join(qa_texts) or "（无问答信息）"

            block_text = (
                f"【教师信息 {idx}】\n"
                f"标题：{block_title}\n"
                f"概括：{block_summary}\n"
                f"问答：\n{block_qas}"
            )
            teacher_blocks.append(block_text)

        teacher_info_all = "\n\n".join(teacher_blocks)

        prompt = (
            f"这是历史教材的第 {page_num} 页图片，请先对图片做 OCR，"
            f"提取有意义的段落，并按给定 JSON schema 返回 Paragraph 列表。\n"
            f"下面是教师用书中本页的所有提示信息（可能包含多段问题与参考答案），供你参考理解版面：\n"
            f"{teacher_info_all}\n\n"
            "请注意：\n"
            "1. 一切以图片上的实际文字为准，不要凭空编造内容；\n"
            "2. 如果图片上能找到和教师用书问题对应的位置，可以优先保证这些段落的 OCR 准确；\n"
            "3. 最终只按 Paragraph 的 JSON schema 返回一组段落。\n\n"
            f"JSON schema:\n{schema}\n\n"
            "<__media__>"
        )
    else:
        prompt = (
            f"这是历史教材的第 {page_num} 页图片，请对图片做 OCR，"
            f"提取有意义的段落，并按给定 JSON schema 返回 Paragraph 列表。\n"
            "只根据图片上的文字，不要自己编内容。\n\n"
            f"JSON schema:\n{schema}\n\n"
            "<__media__>"
        )

    # 使用带重试的封装调用大模型
    result = await completion_with_retry(
        messages=[
            {
                "role": "user",
                "content": prompt,
                "medias": {
                    0: {"type": "image", "content": img_b64},
                },
            }
        ],
        json_schema=schema,
        model_filter="name == Qwen/Qwen3-Omni-30B-A3B-Instruct",
    )

    data = orjson.loads(result["text"])  # type: ignore
    paragraphs = [Paragraph.model_validate(item) for item in data]
    print(f"第 {page_num} 页 OCR 完成，段落数：{len(paragraphs)}")

    return page_num, paragraphs


async def ocr_pdf_with_teacher(
    pdf_path: str | Path,
    teacher_chunks: Dict[int, List[TeacherChunk]] | None = None,
    batch_size: int = 4,
    save_path: str | Path | None = None,
    pages: Optional[List[int]] = None,
) -> Dict[int, List[Paragraph]]:
    """
    整本 PDF OCR：
      - teacher_chunks: {页码: [TeacherChunk,...]}
      - batch_size: 每批并发多少页
      - save_path: 如果不为 None，则在每页识别完成后立即把当前所有结果写入该 JSON 文件
      - pages: 可选，只识别指定页码列表（1-based）。如果为 None，则识别整本书。
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(str(pdf_path))

    schema = TypeAdapter(list[Paragraph]).json_schema()
    results: Dict[int, List[Paragraph]] = {}

    # 页码列表（1-based）
    if pages is None:
        page_nums = list(range(1, doc.page_count + 1))
    else:
        # 过滤掉超出范围的页，为了安全起见
        page_nums = [p for p in pages if 1 <= p <= doc.page_count]

    print(f"本次 OCR 处理页码: {page_nums}")

    # 按页码列表分批，并发处理
    for start in range(0, len(page_nums), batch_size):
        batch_page_nums = page_nums[start : start + batch_size]
        tasks = []

        for page_num in batch_page_nums:
            page_index = page_num - 1  # fitz 是 0-based
            page = doc[page_index]
            pix = page.get_pixmap(dpi=200)
            img_bytes = pix.tobytes("png")

            tc_list = teacher_chunks.get(page_num) if teacher_chunks else None
            tasks.append(ocr_one_page(page_num, img_bytes, tc_list, schema))

        batch_results = await asyncio.gather(*tasks, return_exceptions=False)

        # 合并结果，并按需立即保存
        for page_num, paragraphs in batch_results:
            results[page_num] = paragraphs
            print(f"合并第 {page_num} 页结果")

            if save_path is not None:
                save_ocr_result_to_json(results, save_path)
                print(f"已写入当前进度到 {save_path}")

    doc.close()
    return results


def load_teacher_chunks_from_multiple_json(json_paths: List[str]) -> Dict[int, List[TeacherChunk]]:
    """
    从多个教师教材 JSON 文件中加载教师信息，并合并。
    """
    all_chunks: Dict[int, List[TeacherChunk]] = {}

    for json_path in json_paths:
        if not os.path.exists(json_path):
            print(f"文件 {json_path} 不存在，跳过...")
            continue

        with open(json_path, "rb") as f:
            data = orjson.loads(f.read())

        for item in data:
            chunk = TeacherChunk(**item)
            if chunk.page_num is None:
                continue
            # 聚合教师信息：根据页码聚合所有的 TeacherChunk
            all_chunks.setdefault(chunk.page_num, []).append(chunk)

    return all_chunks



def save_ocr_result_to_json(
    ocr_result: Dict[int, List[Paragraph]],
    out_path: str | Path,
):
    """
    把 {页码: [Paragraph,...]} 写到一个 JSON 文件中。
    每次调用都会重写一遍当前结果，方便中途断点恢复。
    """
    # 把 Pydantic 对象转成 Python dict，并且把页码转成字符串 key
    serializable = {
        str(page_num): [para.model_dump() for para in paragraphs]
        for page_num, paragraphs in ocr_result.items()
    }

    out_path = Path(out_path)
    with out_path.open("wb") as f:
        f.write(orjson.dumps(serializable, option=orjson.OPT_INDENT_2))



# ========= 原来的整本 demo（可留可删） =========

async def demo_pdf_with_teacher():
    pdf_path = r"D:\Code\thinkthinksyn\thinkthinksyn\history_rag\data\stu_book\第1册_re.pdf"
    teacher_json_paths = [
        r"D:\Code\thinkthinksyn\thinkthinksyn\history_rag\teacher_chunks_ch1.json",
        r"D:\Code\thinkthinksyn\thinkthinksyn\history_rag\teacher_chunks_ch2.json",
        r"D:\Code\thinkthinksyn\thinkthinksyn\history_rag\teacher_chunks_ch3.json",
        r"D:\Code\thinkthinksyn\thinkthinksyn\history_rag\teacher_chunks_ch4.json"
    ]
    out_json = r"D:\Code\thinkthinksyn\thinkthinksyn\history_rag\第一册_ocr.json"

    teacher_chunks = load_teacher_chunks_from_multiple_json(teacher_json_paths)

    ocr_result = await ocr_pdf_with_teacher(
        pdf_path,
        teacher_chunks=teacher_chunks,
        batch_size=4,
        save_path=out_json,   # 整本书边跑边保存
        pages=None,           # None 表示整本书
    )

    print("总页数:", len(ocr_result))


# ========= 新增：只跑 4 页的 demo，用于验证 =========

async def demo_pdf_with_teacher_4pages():
    pdf_path = r"D:\Code\thinkthinksyn\thinkthinksyn\history_rag\data\stu_book\第1册_re.pdf"
    teacher_json_paths = [
        r"D:\Code\thinkthinksyn\thinkthinksyn\history_rag\teacher_chunks_ch1.json",
        r"D:\Code\thinkthinksyn\thinkthinksyn\history_rag\teacher_chunks_ch2.json",
        r"D:\Code\thinkthinksyn\thinkthinksyn\history_rag\teacher_chunks_ch3.json",
        r"D:\Code\thinkthinksyn\thinkthinksyn\history_rag\teacher_chunks_ch4.json"
    ]
    out_json = r"D:\Code\thinkthinksyn\thinkthinksyn\history_rag\第一册_ocr_demo_4pages.json"

    # 加载多个教师教材 JSON 文件并合并
    teacher_chunks = load_teacher_chunks_from_multiple_json(teacher_json_paths)

    # 只跑第 1~4 页（你可以修改为任何页码）
    pages = [1, 2, 3, 4]

    ocr_result = await ocr_pdf_with_teacher(
        pdf_path,
        teacher_chunks=teacher_chunks,
        batch_size=4,      # 每批 4 页
        save_path=out_json,
        pages=pages,       # 只处理指定页码
    )

    print("本次 demo 实际处理的页码:", sorted(ocr_result.keys()))
    for page_num in sorted(ocr_result.keys()):
        print(f"==== 第 {page_num} 页示例 ====")
        for para in ocr_result[page_num]:
            print("标题:", para.title)
            preview = para.content.replace("\n", " ")
            if len(preview) > 80:
                preview = preview[:80] + "..."
            print("内容:", preview)
            print("------")


if __name__ == "__main__":
    # 只想验证流程时，跑 4 页 demo：
    asyncio.run(demo_pdf_with_teacher_4pages())

    # 如果以后要整本跑，再改成：
    # asyncio.run(demo_pdf_with_teacher())
