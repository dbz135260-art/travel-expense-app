import streamlit as st
import json
import io
import zipfile
from datetime import datetime, date
from typing import List, Dict
import fitz
import requests

st.set_page_config(
    page_title="差旅补助计算器",
    page_icon="🧾",
    layout="wide"
)

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

SYSTEM_PROMPT = """你是一个差旅单据处理助手。从上传的PDF行程单中提取信息。

每个文件是一个航空电子客票行程单。对每个文件提取以下字段：
1. 姓名 - 旅客姓名
2. 日期 - 出发日期（格式：YYYY-MM-DD）
3. 票面金额 - 电子客票行程单上的"合计"金额（数字，单位：元）
4. 类型 - 判断这张票是"出发"还是"返程"还是"中转"

类型判断规则：
- 从常住地出发 → "出发"
- 返回常住地 → "返程"
- 两个非常住地之间的行程 → "中转"

如果你不确定常住地，根据行程判断：去程为"出发"，回程为"返程"。

输出纯JSON数组，不要markdown标记：
[
  {
    "姓名": "刘延川",
    "日期": "2026-04-15",
    "票面金额": 880.00,
    "类型": "出发",
    "文件名": "行程单_刘延川 04月15日 大连-郑州_xxx.pdf"
  }
]

只提取原始数据，不要计算任何补助。"""

USER_PROMPT_TEMPLATE = """项目名称：{project_name}

以下是所有PDF文件提取出的文字内容，请提取信息：

{pdf_texts}

返回JSON数组。"""


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


def calc_subsidy(items: List[Dict], surname_prefix: str) -> Dict:
    """Calculate subsidies from extracted items, return processed results"""
    # Group by person
    persons = {}
    for item in items:
        name = item.get("姓名", "未知")
        if name not in persons:
            persons[name] = []
        persons[name].append(item)

    results = []
    total_fare = 0
    total_base = 0
    total_extra = 0
    total_all = 0

    for person_name, tickets in persons.items():
        # Sort by date
        tickets.sort(key=lambda x: x.get("日期", ""))

        dates = []
        departures = 0
        returns = 0
        transfers = 0

        for t in tickets:
            try:
                d = datetime.strptime(str(t.get("日期", "")), "%Y-%m-%d").date()
                dates.append(d)
            except:
                pass
            tp = t.get("类型", "")
            if tp == "出发":
                departures += 1
            elif tp == "返程":
                returns += 1
            else:
                transfers += 1

        if not dates:
            continue

        min_date = min(dates)
        max_date = max(dates)
        travel_days = (max_date - min_date).days + 1

        base_subsidy = travel_days * 100
        extra_subsidy = (departures + returns + transfers) * 80
        total_person = base_subsidy + extra_subsidy

        for t in tickets:
            fare = float(t.get("票面金额", 0))
            d_str = str(t.get("日期", ""))
            # Try to convert date to display format like "4月15日"
            try:
                dt = datetime.strptime(d_str, "%Y-%m-%d")
                display_date = f"{dt.month}月{dt.day}日"
            except:
                display_date = d_str

            new_filename = f"{surname_prefix} {person_name} {display_date} {fare:.2f}.pdf"

            results.append({
                "姓名": person_name,
                "日期": d_str,
                "票面金额": fare,
                "基础补助": base_subsidy,
                "额外补助": extra_subsidy,
                "总金额": total_person,
                "原文件名": t.get("文件名", ""),
                "新文件名": new_filename
            })

            total_fare += fare

        total_base += base_subsidy
        total_extra += extra_subsidy
        total_all += total_person

    return {
        "items": results,
        "total_fare": total_fare,
        "total_base": total_base,
        "total_extra": total_extra,
        "total_all": total_all
    }


def main():
    st.title("🧾 差旅补助计算器")
    st.markdown("上传PDF行程单，自动提取信息、计算补助、生成改名文件")

    # Sidebar
    with st.sidebar:
        st.header("配置")
        # Load from Streamlit secrets if configured
        default_key = st.secrets.get("DEEPSEEK_API_KEY", "")
        api_key = st.text_input(
            "DeepSeek API Key",
            type="password",
            value=default_key,
            help="已配置默认Key则自动填入，可覆盖。https://platform.deepseek.com/ 获取"
        )
        project_name = st.text_input("项目名称", value="默认项目")
        surname_prefix = st.text_input("姓氏前缀", value="段",
                                       help="文件名前缀，如'段' → '段 刘延川 4月15日 880.00.pdf'")
        st.markdown("---")
        st.markdown("### 补助规则")
        st.markdown("""
- 基础补助：100元/天
- 额外补助：80元/次（每次出发/返程/中转）
- 总金额 = 基础补助 + 额外补助
- 行程天数 = 最晚日期 - 最早日期 + 1
        """)
        st.markdown("---")
        st.markdown("### 改名规则")
        st.markdown("`{姓氏前缀} {姓名} {月}日 {金额}.pdf`")

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

    col1, col2 = st.columns([1, 5])
    with col1:
        process_btn = st.button("🚀 开始处理", type="primary", use_container_width=True)

    if not process_btn:
        st.stop()

    if not api_key:
        st.error("请在左侧输入 DeepSeek API Key")
        st.stop()

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

    # Step 2: Call DeepSeek (extract only, no calculation)
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

    progress_bar.progress(0.8, text="正在计算补助...")

    # Step 3: Parse result
    try:
        data = parse_result(raw_response)
    except json.JSONDecodeError as e:
        st.error(f"解析LLM返回结果失败: {str(e)}")
        st.markdown("**原始返回内容：**")
        st.code(raw_response[:3000])
        st.stop()

    # Step 4: Python does the calculation
    result = calc_subsidy(data, surname_prefix)

    progress_bar.progress(1.0, text="完成！")
    status_text.text("✅ 处理完成")

    # Display results
    st.header("📊 处理结果")

    # Detail table
    st.subheader("明细")
    table_data = []
    rename_map = {}
    for item in result["items"]:
        table_data.append({
            "姓名": item["姓名"],
            "日期": item["日期"],
            "票面金额": item["票面金额"],
            "基础补助": item["基础补助"],
            "额外补助": item["额外补助"],
            "总金额": item["总金额"],
            "新文件名": item["新文件名"]
        })
        orig = item["原文件名"]
        new = item["新文件名"]
        if orig and orig in file_map:
            rename_map[orig] = new

    st.dataframe(table_data, use_container_width=True, hide_index=True)

    # Summary
    st.subheader("💰 总计")
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("票面金额合计", f"¥{result['total_fare']:.2f}")
    sc2.metric("基础补助合计", f"¥{result['total_base']:.2f}")
    sc3.metric("额外补助合计", f"¥{result['total_extra']:.2f}")
    sc4.metric("总金额合计", f"¥{result['total_all']:.2f}")

    # Rename mapping + download
    st.header("📁 文件改名")
    if rename_map:
        rename_table = []
        for orig, new in rename_map.items():
            rename_table.append({"原文件名": orig, "新文件名": new})

        if rename_table:
            st.dataframe(rename_table, use_container_width=True, hide_index=True)

            zip_bytes = create_renamed_zip(file_map, rename_map)
            st.download_button(
                label="📦 下载改名后的文件（ZIP）",
                data=zip_bytes,
                file_name=f"{project_name}_改名文件.zip",
                mime="application/zip",
                type="primary"
            )

    # Raw API response
    with st.expander("查看API原始返回"):
        st.code(raw_response[:3000])

    # Extracted text
    with st.expander("查看PDF提取的原始文本"):
        st.text(combined_text[:5000])


if __name__ == "__main__":
    main()
