import streamlit as st
import json
import io
import zipfile
import time
from typing import List, Dict
import fitz
import requests

st.set_page_config(
    page_title="差旅补助计算器",
    page_icon="🧾",
    layout="wide"
)

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

SYSTEM_PROMPT = """你是一个差旅单据处理助手。从上传的PDF行程单中提取信息并计算补助。

每个文件是一个差旅行程单。请严格按以下规则处理：

## 提取字段
1. 姓名
2. 行程日期（出发日期，格式：YYYY-MM-DD）
3. 票面金额（交通费用，数字，单位：元）

## 补助计算规则
- 基础补助 = 100元/天 × 行程天数
- 额外补助 = 出发日+80元 + 返程日+80元 + 中转日+80元（如有中转）
- 总金额 = 基础补助 + 额外补助（不含票面金额）

## 输出格式
必须返回纯JSON数组，不要包含其他任何文字或markdown标记：
[
  {
    "姓名": "张三",
    "日期": "2024-03-15",
    "票面金额": 538.00,
    "基础补助": 200.00,
    "额外补助": 160.00,
    "总金额": 360.00,
    "文件名": "张三_2024-03-15_538.00.pdf"
  }
]

最后一条记录必须是一个汇总行：
  {
    "姓名": "合计",
    "日期": "",
    "票面金额": <所有票面金额总和>,
    "基础补助": <所有基础补助总和>,
    "额外补助": <所有额外补助总和>,
    "总金额": <所有总金额总和>,
    "文件名": ""
  }"""

USER_PROMPT_TEMPLATE = """项目名称：{project_name}

以下是所有PDF文件提取出的文字内容，请按规则处理：

{pdf_texts}

请返回JSON数组，包含每个人的明细和最后一条合计行。"""


def extract_pdf_text(file_bytes: bytes) -> str:
    """Extract text from PDF using PyMuPDF"""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text_parts = []
        for page in doc:
            text = page.get_text().strip()
            if text:
                text_parts.append(text)
        doc.close()
        return "\n".join(text_parts)
    except Exception as e:
        return f"[PDF解析错误: {str(e)}]"


def call_deepseek(api_key: str, system_prompt: str, user_prompt: str) -> str:
    """Call DeepSeek API"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.01,
        "max_tokens": 4096
    }

    resp = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    result = resp.json()
    return result["choices"][0]["message"]["content"]


def parse_result(content: str) -> List[Dict]:
    """Parse JSON from LLM response, handling markdown fences"""
    text = content.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        text = "\n".join(lines).strip()
    return json.loads(text)


def create_renamed_zip(original_files: Dict[str, bytes], rename_map: Dict[str, str]) -> bytes:
    """Create ZIP with renamed files"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for orig_name, file_bytes in original_files.items():
            new_name = rename_map.get(orig_name, orig_name)
            zf.writestr(new_name, file_bytes)
    buf.seek(0)
    return buf.getvalue()


def main():
    st.title("🧾 差旅补助计算器")
    st.markdown("上传PDF行程单，自动提取信息、计算补助、生成改名文件")

    # Sidebar
    with st.sidebar:
        st.header("配置")
        api_key = st.text_input("DeepSeek API Key", type="password",
                                help="https://platform.deepseek.com/ 获取")
        project_name = st.text_input("项目名称", value="默认项目")
        st.markdown("---")
        st.markdown("### 补助规则")
        st.markdown("""
- 基础补助：100元/天
- 额外补助：
  - 出发日：+80元
  - 返程日：+80元
  - 中转日：+80元
- 总金额 = 基础补助 + 额外补助
        """)
        st.markdown("---")
        st.markdown("### 改名规则")
        st.markdown("`姓名_日期_票面金额.pdf`")

    # Main area
    uploaded_files = st.file_uploader(
        "上传PDF行程单（可多选）",
        type=["pdf"],
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.info("请上传PDF文件开始处理")
        st.stop()

    st.success(f"已上传 {len(uploaded_files)} 个文件")

    # Process button
    col1, col2 = st.columns([1, 5])
    with col1:
        process_btn = st.button("🚀 开始处理", type="primary", use_container_width=True)

    if not process_btn:
        st.stop()

    if not api_key:
        st.error("请在左侧输入 DeepSeek API Key")
        st.stop()

    # Progress
    progress_bar = st.progress(0, text="解析PDF中...")
    status_text = st.empty()

    # Step 1: Extract PDF text
    status_text.text("正在解析PDF文件...")
    pdf_texts = []
    file_map = {}
    for i, f in enumerate(uploaded_files):
        bytes_data = f.read()
        file_map[f.name] = bytes_data
        text = extract_pdf_text(bytes_data)
        pdf_texts.append(f"【文件{i+1}: {f.name}】\n{text}")
        progress_bar.progress((i + 1) / (len(uploaded_files) + 2))

    combined_text = "\n\n---\n\n".join(pdf_texts)

    # Step 2: Call DeepSeek
    status_text.text("正在调用 DeepSeek API...")
    progress_bar.progress(0.6, text="正在调用 DeepSeek API...")

    user_prompt = USER_PROMPT_TEMPLATE.format(
        project_name=project_name,
        pdf_texts=combined_text
    )

    try:
        raw_response = call_deepseek(api_key, SYSTEM_PROMPT, user_prompt)
    except Exception as e:
        st.error(f"API调用失败: {str(e)}")
        st.stop()

    progress_bar.progress(0.8, text="正在解析结果...")

    # Step 3: Parse result
    try:
        data = parse_result(raw_response)
    except json.JSONDecodeError as e:
        st.error(f"解析LLM返回结果失败: {str(e)}")
        st.markdown("**原始返回内容：**")
        st.code(raw_response[:2000])
        st.stop()

    # Separate summary row
    summary = data[-1] if data and data[-1].get("姓名") == "合计" else None
    items = data[:-1] if summary else data

    progress_bar.progress(1.0, text="完成！")
    status_text.text("✅ 处理完成")

    # Display results
    st.header("📊 处理结果")

    # Detail table
    st.subheader("明细")
    table_data = []
    rename_map = {}
    for item in items:
        table_data.append({
            "姓名": item.get("姓名", ""),
            "日期": item.get("日期", ""),
            "票面金额": item.get("票面金额", 0),
            "基础补助": item.get("基础补助", 0),
            "额外补助": item.get("额外补助", 0),
            "总金额": item.get("总金额", 0),
        })
        fname = item.get("文件名", "")
        if fname:
            rename_map[fname] = fname  # new name

    # Map original -> new names
    for i, item in enumerate(items):
        new_name = item.get("文件名", "")
        if new_name and i < len(uploaded_files):
            rename_map[uploaded_files[i].name] = new_name

    st.dataframe(table_data, use_container_width=True, hide_index=True)

    # Summary
    if summary:
        st.subheader("💰 总计")
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("票面金额合计", f"¥{summary.get('票面金额', 0):.2f}")
        sc2.metric("基础补助合计", f"¥{summary.get('基础补助', 0):.2f}")
        sc3.metric("额外补助合计", f"¥{summary.get('额外补助', 0):.2f}")
        sc4.metric("总金额合计", f"¥{summary.get('总金额', 0):.2f}")

    # Rename mapping + download
    st.header("📁 文件改名")
    if rename_map:
        rename_table = []
        for orig, new in rename_map.items():
            if orig in file_map:
                rename_table.append({"原文件名": orig, "新文件名": new})

        if rename_table:
            st.dataframe(rename_table, use_container_width=True, hide_index=True)

            # Create ZIP with renamed files
            zip_bytes = create_renamed_zip(file_map, rename_map)
            st.download_button(
                label="📦 下载改名后的文件（ZIP）",
                data=zip_bytes,
                file_name=f"{project_name}_改名文件.zip",
                mime="application/zip",
                type="primary"
            )

    # Show raw API response (expandable)
    with st.expander("查看API原始返回"):
        st.code(raw_response[:3000])

    # Show extracted PDF text (expandable)
    with st.expander("查看PDF提取的原始文本"):
        st.text(combined_text[:5000])


if __name__ == "__main__":
    main()
