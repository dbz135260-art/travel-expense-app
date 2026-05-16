import streamlit as st
import json
import io
import zipfile
import re
import xml.etree.ElementTree as ET
from datetime import datetime, date
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import fitz
import requests

st.set_page_config(page_title="差旅/劳务费处理", page_icon="🧾", layout="wide")

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# ─── OFD Parser ────────────────────────────────────────

def parse_ofd(file_bytes: bytes) -> Dict:
    """Parse OFD (数电票) file - extract structured data from attachment XML"""
    import zipfile
    result = {"姓名": "", "金额": 0.0, "日期": "", "保险费": 0.0, "类型": "ofd", "原始文本": ""}
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
            # Find XML files
            xml_files = [n for n in z.namelist() if n.endswith(".xml") and "Attachs" in n]
            for xml_path in xml_files:
                with z.open(xml_path) as f:
                    raw = f.read()
                    content = raw.decode("utf-8", errors="replace")
                    result["原始文本"] += content + "\n"

            # Also check Document.xml for any text
            doc_xmls = [n for n in z.namelist() if n.endswith("Document.xml")]
            for xml_path in doc_xmls:
                with z.open(xml_path) as f:
                    raw = f.read()
                    result["原始文本"] += raw.decode("utf-8", errors="replace") + "\n"

            # Parse XBRL data from attachment XMLs
            for xml_path in xml_files:
                with z.open(xml_path) as f:
                    raw = f.read()
                    try:
                        root = ET.fromstring(raw)
                        # Namespace handling
                        ns = {}
                        for m in re.finditer(r'xmlns:(\w+)="([^"]+)"', raw.decode("utf-8", errors="replace")):
                            ns[m.group(1)] = m.group(2)

                        # Extract passenger name
                        for tag in ["atr:PassengerName", "PassengerName"]:
                            el = root.find(f".//{tag}", ns) if tag in ns.get("atr", "") or ":" not in tag else root.find(f".//{tag}")
                            if el is not None and el.text:
                                result["姓名"] = el.text.strip()
                                break

                        # Try to find name in non-namespace way
                        if not result["姓名"]:
                            for el in root.iter():
                                if "PassengerName" in el.tag and el.text:
                                    result["姓名"] = el.text.strip()
                                    break

                        # Extract total amount
                        for tag in ["atr:TotalAmount", "TotalAmount"]:
                            el = root.find(f".//{tag}", ns) if tag in ns.get("atr", "") or ":" not in tag else root.find(f".//{tag}")
                            if el is not None and el.text:
                                result["金额"] = float(el.text)
                                break
                        if result["金额"] == 0:
                            for el in root.iter():
                                if "TotalAmount" in el.tag and el.text:
                                    try:
                                        result["金额"] = float(el.text)
                                    except: pass
                                    break

                        # Extract insurance (if present)
                        for tag in ["atr:InsuranceFee", "InsuranceFee",
                                     "atr:OtherTaxes", "OtherTaxes"]:
                            el = root.find(f".//{tag}", ns) if tag in ns.get("atr", "") or ":" not in tag else root.find(f".//{tag}")
                            if el is not None and el.text:
                                try:
                                    val = float(el.text)
                                    # Check if it's labeled as insurance or other
                                    if "Insurance" in tag or "保险" in str(raw):
                                        result["保险费"] = val
                                except: pass

                        # Extract date
                        for tag in ["atr:CarrierDate", "CarrierDate",
                                     "atr:IssuanceDate", "IssuanceDate"]:
                            el = root.find(f".//{tag}", ns) if tag in ns.get("atr", "") or ":" not in tag else root.find(f".//{tag}")
                            if el is not None and el.text:
                                result["日期"] = el.text.strip()
                                break
                        if not result["日期"]:
                            for el in root.iter():
                                if "CarrierDate" in el.tag and el.text:
                                    result["日期"] = el.text.strip()
                                    break

                    except ET.ParseError:
                        continue
    except Exception as e:
        result["原始文本"] = f"OFD解析错误: {str(e)}"
    return result


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


def call_deepseek(api_key: str, system_prompt: str, user_prompt: str, max_tokens: int = 4096) -> str:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.01,
        "max_tokens": max_tokens
    }
    resp = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def parse_json_response(content: str) -> List[Dict]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        text = "\n".join(lines).strip()
    return json.loads(text)


def create_renamed_zip(file_map: Dict[str, bytes], rename_map: Dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for orig_name, file_bytes in file_map.items():
            new_name = rename_map.get(orig_name, orig_name)
            zf.writestr(new_name, file_bytes)
    buf.seek(0)
    return buf.getvalue()


def extract_name_from_filename(fname: str) -> str:
    """Extract teacher name from filename. Try patterns like 姓名_xxx, xxx-姓名, etc."""
    # Strip extension
    name = Path(fname).stem
    # Common patterns in Chinese filenames with names
    # If filename starts with Chinese name (2-4 chars)
    # Try to find a known pattern
    return name


# ─── Parse pasted teacher info ──────────────────────

def parse_teacher_info(text: str) -> List[Dict]:
    """Parse pasted teacher info (name + bank card or more fields).
    Accepts: tab-separated, space-separated, table format, etc."""
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    teachers = []

    for line in lines:
        # Try tab separation first
        parts = re.split(r"[\t,，、\s]{1,}", line)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) >= 2:
            # First part should be name (Chinese chars), second should be bank card (digits)
            name = parts[0]
            # Find the first all-digit field (bank card)
            card = ""
            phone = ""
            id_num = ""
            for p in parts[1:]:
                if re.match(r"^\d{15,}$", p):  # bank card (15+ digits)
                    card = p
                elif re.match(r"^\d{11}$", p):  # phone
                    phone = p
                elif re.match(r"^\d{17}[\dXx]$", p):  # ID
                    id_num = p
                elif re.match(r"^\d{18}$", p):
                    id_num = p

            teacher = {"姓名": name, "银行卡号": card, "手机号": phone, "身份证号": id_num}
            if card:  # Only add if we found a bank card
                teachers.append(teacher)

    return teachers


# ─── Excel generators ──────────────────────────────

def generate_travel_excel(teachers: List[Dict], project_name: str, project_code: str) -> bytes:
    """Generate 差旅费 Excel matching template format"""
    import openpyxl
    from openpyxl.styles import Font, Alignment, Border, Side

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "差旅费发放表"

    thin_border = Border(
        left=Side(style='thin'), right=Side(style='thin'),
        top=Side(style='thin'), bottom=Side(style='thin')
    )
    center = Alignment(horizontal='center', vertical='center')
    left_align = Alignment(horizontal='left', vertical='center')
    title_font = Font(name='宋体', bold=True, size=14)
    header_font = Font(name='宋体', bold=True, size=11)
    normal_font = Font(name='宋体', size=11)

    # Row 1: Title
    ws.merge_cells('A1:D1')
    ws['A1'] = f"差旅费发放表"
    ws['A1'].font = title_font
    ws['A1'].alignment = center

    # Row 2: Project name
    ws.merge_cells('A2:D2')
    ws['A2'] = f"项目名称：{project_name}"
    ws['A2'].font = Font(name='宋体', bold=True, size=11)
    ws['A2'].alignment = left_align

    # Row 3: Headers
    headers = ["序号", "姓名", "银行借记卡号", "实发金额"]
    for col, h in enumerate(headers, 1):
        cell = ws.cell(row=3, column=col, value=h)
        cell.font = header_font
        cell.alignment = center
        cell.border = thin_border

    # Data rows
    total_amount = 0
    for i, t in enumerate(teachers, 1):
        ws.cell(row=3+i, column=1, value=i).font = normal_font
        ws.cell(row=3+i, column=1).alignment = center
        ws.cell(row=3+i, column=1).border = thin_border

        ws.cell(row=3+i, column=2, value=t.get("姓名", "")).font = normal_font
        ws.cell(row=3+i, column=2).alignment = center
        ws.cell(row=3+i, column=2).border = thin_border

        ws.cell(row=3+i, column=3, value=t.get("银行卡号", "")).font = normal_font
        ws.cell(row=3+i, column=3).alignment = center
        ws.cell(row=3+i, column=3).border = thin_border

        amt = float(t.get("实发金额", 0))
        ws.cell(row=3+i, column=4, value=amt).font = normal_font
        ws.cell(row=3+i, column=4).alignment = center
        ws.cell(row=3+i, column=4).border = thin_border
        ws.cell(row=3+i, column=4).number_format = '#,##0.00'
        total_amount += amt

    # Add detailed amounts columns after 实发金额 (column 4)
    # Check how many detail columns we need
    max_details = max((len(t.get("明细金额", [])) for t in teachers), default=0)
    if max_details > 0:
        for col_idx in range(max_details):
            col_letter = openpyxl.utils.get_column_letter(5 + col_idx)
            ws.cell(row=3, column=5+col_idx, value=f"票面{col_idx+1}").font = header_font
            ws.cell(row=3, column=5+col_idx).alignment = center
            ws.cell(row=3, column=5+col_idx).border = thin_border

        # Re-merge title to cover all columns
        ws.merge_cells(f'A1:{openpyxl.utils.get_column_letter(4+max_details)}1')
        ws.merge_cells(f'A2:{openpyxl.utils.get_column_letter(4+max_details)}2')

        # Fill detail amounts per row
        for i, t in enumerate(teachers, 1):
            details = t.get("明细金额", [])
            for j, amt in enumerate(details):
                cell = ws.cell(row=3+i, column=5+j, value=float(amt))
                cell.font = normal_font
                cell.alignment = center
                cell.border = thin_border
                cell.number_format = '#,##0.00'

    # Summary row
    summary_row = 4 + len(teachers)
    ws.merge_cells(f'A{summary_row}:C{summary_row}')
    ws.cell(row=summary_row, column=1, value="合    计").font = Font(name='宋体', bold=True, size=11)
    ws.cell(row=summary_row, column=1).alignment = center
    ws.cell(row=summary_row, column=1).border = thin_border
    for c in range(2, 5):
        ws.cell(row=summary_row, column=c).border = thin_border
    total_cell = ws.cell(row=summary_row, column=4, value=total_amount)
    total_cell.font = Font(name='宋体', bold=True, size=11)
    total_cell.alignment = center
    total_cell.number_format = '#,##0.00'
    total_cell.border = thin_border

    # Column widths
    ws.column_dimensions['A'].width = 8
    ws.column_dimensions['B'].width = 12
    ws.column_dimensions['C'].width = 24
    ws.column_dimensions['D'].width = 14
    for col_idx in range(max_details):
        ws.column_dimensions[openpyxl.utils.get_column_letter(5+col_idx)].width = 12

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf.getvalue()


def generate_labor_excel(teachers: List[Dict], project_name: str, project_code: str) -> bytes:
    """Generate 劳务费 Excel matching template format"""
    import openpyxl
    from openpyxl.styles import Font, Alignment, Border, Side

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "劳务费发放表"

    thin_border = Border(
        left=Side(style='thin'), right=Side(style='thin'),
        top=Side(style='thin'), bottom=Side(style='thin')
    )
    center = Alignment(horizontal='center', vertical='center', wrap_text=True)
    left_align = Alignment(horizontal='left', vertical='center')
    title_font = Font(name='宋体', bold=True, size=14)
    header_font = Font(name='宋体', bold=True, size=10)
    normal_font = Font(name='宋体', size=10)

    # Row 1: Title
    ws.merge_cells('A1:J1')
    ws['A1'] = f"劳务费发放表"
    ws['A1'].font = title_font
    ws['A1'].alignment = center

    # Row 2: Project name and code
    ws.merge_cells('A2:G2')
    ws['A2'] = f"项目名称：{project_name}"
    ws['A2'].font = Font(name='宋体', bold=True, size=10)
    ws['A2'].alignment = left_align
    ws.merge_cells('H2:J2')
    ws['H2'] = f"项目号：{project_code}"
    ws['H2'].font = Font(name='宋体', bold=True, size=10)
    ws['H2'].alignment = left_align

    # Row 3: Headers
    headers = ["序号", "项目执行时间", "工作内容", "手机号", "姓名", "身份证号", "银行卡号", "标准(元/学时)", "学时", "实发金额"]
    for col, h in enumerate(headers, 1):
        cell = ws.cell(row=3, column=col, value=h)
        cell.font = header_font
        cell.alignment = center
        cell.border = thin_border

    # Data rows
    total_amount = 0
    for i, t in enumerate(teachers, 1):
        row_num = 3 + i
        ws.cell(row=row_num, column=1, value=i).font = normal_font
        ws.cell(row=row_num, column=1).alignment = center
        ws.cell(row=row_num, column=1).border = thin_border

        ws.cell(row=row_num, column=2, value=t.get("执行时间", "")).font = normal_font
        ws.cell(row=row_num, column=2).alignment = center
        ws.cell(row=row_num, column=2).border = thin_border

        ws.cell(row=row_num, column=3, value=t.get("工作内容", "讲课")).font = normal_font
        ws.cell(row=row_num, column=3).alignment = center
        ws.cell(row=row_num, column=3).border = thin_border

        ws.cell(row=row_num, column=4, value=t.get("手机号", "")).font = normal_font
        ws.cell(row=row_num, column=4).alignment = center
        ws.cell(row=row_num, column=4).border = thin_border

        ws.cell(row=row_num, column=5, value=t.get("姓名", "")).font = normal_font
        ws.cell(row=row_num, column=5).alignment = center
        ws.cell(row=row_num, column=5).border = thin_border

        ws.cell(row=row_num, column=6, value=t.get("身份证号", "")).font = normal_font
        ws.cell(row=row_num, column=6).alignment = center
        ws.cell(row=row_num, column=6).border = thin_border

        ws.cell(row=row_num, column=7, value=t.get("银行卡号", "")).font = normal_font
        ws.cell(row=row_num, column=7).alignment = center
        ws.cell(row=row_num, column=7).border = thin_border

        rate = float(t.get("标准", 0))
        hours = float(t.get("学时", 0))
        ws.cell(row=row_num, column=8, value=rate).font = normal_font
        ws.cell(row=row_num, column=8).alignment = center
        ws.cell(row=row_num, column=8).border = thin_border
        ws.cell(row=row_num, column=8).number_format = '#,##0.00'

        ws.cell(row=row_num, column=9, value=hours).font = normal_font
        ws.cell(row=row_num, column=9).alignment = center
        ws.cell(row=row_num, column=9).border = thin_border

        amt = rate * hours
        ws.cell(row=row_num, column=10, value=amt).font = normal_font
        ws.cell(row=row_num, column=10).alignment = center
        ws.cell(row=row_num, column=10).border = thin_border
        ws.cell(row=row_num, column=10).number_format = '#,##0.00'
        total_amount += amt

    # Summary row
    summary_row = 4 + len(teachers)
    ws.merge_cells(f'A{summary_row}:G{summary_row}')
    ws.cell(row=summary_row, column=1, value="合    计").font = Font(name='宋体', bold=True, size=10)
    ws.cell(row=summary_row, column=1).alignment = center
    ws.cell(row=summary_row, column=1).border = thin_border
    for c in range(2, 8):
        ws.cell(row=summary_row, column=c).border = thin_border

    total_cell = ws.cell(row=summary_row, column=10, value=total_amount)
    total_cell.font = Font(name='宋体', bold=True, size=10)
    total_cell.alignment = center
    total_cell.number_format = '#,##0.00'
    total_cell.border = thin_border

    # Signature row
    sig_row = summary_row + 1
    ws.merge_cells(f'A{sig_row}:J{sig_row}')
    ws.cell(row=sig_row, column=1, value="  项目负责人：                           财务负责人：                                      组长：                                  ").font = Font(name='宋体', bold=True, size=10)

    # Column widths
    widths = [6, 16, 10, 14, 10, 20, 24, 14, 8, 12]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[openpyxl.utils.get_column_letter(i)].width = w

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf.getvalue()


# ─── LLM prompts ──────────────────────────────────

TRAVEL_EXTRACT_PROMPT = """你是一个票据处理助手。从上传的文件内容中提取信息。

对每个文件提取：
1. 姓名 - 旅客姓名（从文件内容中提取）
2. 日期 - 出发日期（YYYY-MM-DD）
3. 票面金额 - 合计金额（数字，单位：元）
4. 保险费 - 如有保险费用（数字，单位：元，没有则为0）
5. 文件类型 - "飞机行程单" 或 "普通发票"

输出纯JSON数组，不要markdown标记：
[
  {
    "姓名": "刘延川",
    "日期": "2026-04-15",
    "票面金额": 880.00,
    "保险费": 0,
    "文件类型": "飞机行程单",
    "文件名": "原文件名.pdf"
  }
]"""


# ══════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════

def main():
    st.title("🧾 差旅/劳务费处理系统")
    st.markdown("三个模块：差旅补助计算 / 老师差旅报销 / 老师劳务费发放")

    # Sidebar config
    with st.sidebar:
        st.header("配置")
        default_key = st.secrets.get("DEEPSEEK_API_KEY", "")
        api_key = st.text_input("DeepSeek API Key", type="password", value=default_key,
                                help="已配置自动填入，可覆盖")
        project_name = st.text_input("项目名称", value="2025-N4-PX16")
        project_code = st.text_input("项目号", value="2025-N4-PX16",
                                     help="劳务费模板的项目号")

    if not api_key:
        st.warning("请在左侧输入 DeepSeek API Key")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["💰 差旅补助计算", "✈️ 老师差旅报销", "👨‍🏫 老师劳务费发放"])

    # ─── TAB 1: 差旅补助计算（原功能）────────────────
    with tab1:
        render_tab_subsidy(api_key, project_name)

    # ─── TAB 2: 老师差旅报销 ────────────────────
    with tab2:
        render_tab_travel(api_key, project_name, project_code)

    # ─── TAB 3: 老师劳务费发放 ────────────────────
    with tab3:
        render_tab_labor(project_name, project_code)


def render_tab_subsidy(api_key, project_name):
    """原差旅补助计算功能"""
    st.subheader("差旅补助计算")

    uploaded_files = st.file_uploader("上传PDF行程单", type=["pdf"], accept_multiple_files=True,
                                      key="subsidy_files")

    if not uploaded_files:
        st.info("请上传PDF文件")
        return

    st.success(f"已上传 {len(uploaded_files)} 个文件")

    if not st.button("🚀 开始计算", type="primary", key="subsidy_btn"):
        return

    progress_bar = st.progress(0, text="解析PDF中...")
    status_text = st.empty()

    pdf_texts = []
    file_map = {}
    for i, f in enumerate(uploaded_files):
        bytes_data = f.read()
        file_map[f.name] = bytes_data
        text = extract_pdf_text(bytes_data)
        pdf_texts.append(f"【文件{i+1}: {f.name}】\n{text}")
        progress_bar.progress((i + 1) / (len(uploaded_files) + 2))

    combined_text = "\n\n---\n\n".join(pdf_texts)

    status_text.text("调用 DeepSeek API...")
    progress_bar.progress(0.6)

    # LLM extracts raw data
    sys_prompt = """你是一个差旅单据处理助手。从上传的PDF行程单中提取信息，不要计算。

对每个文件提取以下字段：
1. 姓名 - 旅客姓名
2. 日期 - 出发日期（YYYY-MM-DD）
3. 票面金额 - 合计金额（数字，元）
4. 类型 - "出发"或"返程"或"中转"
5. 文件名 - 原文件名

输出纯JSON数组。"""
    user_prompt = f"项目名称：{project_name}\n\n文件内容：\n{combined_text}"

    try:
        raw = call_deepseek(api_key, sys_prompt, user_prompt)
    except Exception as e:
        st.error(f"API调用失败: {e}")
        return

    progress_bar.progress(0.8)

    try:
        data = parse_json_response(raw)
    except Exception as e:
        st.error(f"解析结果失败: {e}")
        st.code(raw[:2000])
        return

    # Python calculates
    persons = {}
    for item in data:
        name = item.get("姓名", "未知")
        if name not in persons:
            persons[name] = []
        persons[name].append(item)

    results = []
    total_fare = 0
    total_base = 0
    total_extra = 0
    total_all = 0
    rename_map = {}

    for person_name, tickets in persons.items():
        tickets.sort(key=lambda x: x.get("日期", ""))
        dates = []
        d_count, r_count, t_count = 0, 0, 0
        for tk in tickets:
            try:
                dates.append(datetime.strptime(str(tk.get("日期", "")), "%Y-%m-%d").date())
            except:
                pass
            tp = tk.get("类型", "")
            if tp == "出发": d_count += 1
            elif tp == "返程": r_count += 1
            else: t_count += 1

        if not dates:
            continue
        travel_days = (max(dates) - min(dates)).days + 1
        base = travel_days * 100
        extra = (d_count + r_count + t_count) * 80
        total_person = base + extra

        for tk in tickets:
            fare = float(tk.get("票面金额", 0))
            d_str = str(tk.get("日期", ""))
            try:
                dt = datetime.strptime(d_str, "%Y-%m-%d")
                disp_date = f"{dt.month}月{dt.day}日"
            except:
                disp_date = d_str
            new_name = f"{person_name} {disp_date} {fare:.2f}.pdf"
            results.append({
                "姓名": person_name, "日期": d_str, "票面金额": fare,
                "基础补助": base, "额外补助": extra, "总金额": total_person,
                "原文件名": tk.get("文件名", ""), "新文件名": new_name
            })
            orig = tk.get("文件名", "")
            if orig and orig in file_map:
                rename_map[orig] = new_name
            total_fare += fare
        total_base += base
        total_extra += extra
        total_all += total_person

    progress_bar.progress(1.0)
    status_text.text("✅ 完成")

    # Display
    st.subheader("明细")
    st.dataframe([{
        "姓名": r["姓名"], "日期": r["日期"], "票面金额": r["票面金额"],
        "基础补助": r["基础补助"], "额外补助": r["额外补助"],
        "总金额": r["总金额"], "新文件名": r["新文件名"]
    } for r in results], use_container_width=True, hide_index=True)

    st.subheader("💰 总计")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("票面金额合计", f"¥{total_fare:.2f}")
    c2.metric("基础补助合计", f"¥{total_base:.2f}")
    c3.metric("额外补助合计", f"¥{total_extra:.2f}")
    c4.metric("总金额合计", f"¥{total_all:.2f}")

    if rename_map:
        st.subheader("📁 文件改名")
        st.dataframe([{"原文件名": k, "新文件名": v} for k, v in rename_map.items()],
                     use_container_width=True, hide_index=True)
        zip_bytes = create_renamed_zip(file_map, rename_map)
        st.download_button("📦 下载改名文件(ZIP)", data=zip_bytes,
                          file_name=f"{project_name}_改名文件.zip", mime="application/zip")


def render_tab_travel(api_key, project_name, project_code):
    """老师差旅报销模块"""
    st.subheader("老师差旅报销")

    st.markdown("""
    **规则说明：**
    - 实发金额 = 行程单/发票票面总金额
    - 飞机行程单含保险费 → 保险费+票面金额
    - 普通发票从文件名提取姓名
    """)

    # Step 1: Upload files
    uploaded_files = st.file_uploader(
        "上传PDF/OFD文件（文件名需含老师姓名）",
        type=["pdf", "ofd"],
        accept_multiple_files=True,
        key="travel_files"
    )

    if not uploaded_files:
        st.info("请上传文件")
        return

    st.success(f"已上传 {len(uploaded_files)} 个文件")

    # Step 2: Paste bank info
    st.markdown("---")
    st.markdown("**老师银行卡信息** — 粘贴表格或文本（姓名 + 银行卡号）")
    bank_text = st.text_area(
        "支持格式：姓名 银行卡号（每行一位老师）",
        placeholder="例：\n刘延川 6222620910068634421\n张伟 9558800200125094454\n王芳 6222620140009990696\n谢维娜 6229478520291044195",
        height=150, key="travel_bank"
    )

    teachers = []
    if bank_text.strip():
        teachers = parse_teacher_info(bank_text.strip())
        if teachers:
            st.success(f"识别到 {len(teachers)} 位老师")
            st.dataframe(teachers, use_container_width=True, hide_index=True)

    if not st.button("🚀 处理并生成Excel", type="primary", key="travel_btn"):
        return

    if not teachers:
        st.warning("请输入老师银行卡信息")
        return

    # Process files
    progress_bar = st.progress(0, text="解析文件中...")
    status_text = st.empty()

    file_data = []
    file_map = {}
    pdf_texts = []
    ofd_data = []

    for i, f in enumerate(uploaded_files):
        bytes_data = f.read()
        file_map[f.name] = bytes_data
        ext = Path(f.name).suffix.lower()

        if ext == ".ofd":
            # Parse OFD directly
            parsed = parse_ofd(bytes_data)
            parsed["文件名"] = f.name
            ofd_data.append(parsed)
            file_data.append({
                "文件名": f.name,
                "姓名": parsed.get("姓名", ""),
                "金额": parsed.get("金额", 0),
                "保险费": parsed.get("保险费", 0),
                "日期": parsed.get("日期", ""),
                "类型": "ofd",
                "原始文本": parsed.get("原始文本", "")
            })
            status_text.text(f"OFD解析完成: {f.name}")
        else:
            text = extract_pdf_text(bytes_data)
            pdf_texts.append(f"【文件{i+1}: {f.name}】\n{text}")
            file_data.append({"文件名": f.name, "类型": "pdf", "原始文本": text})

        progress_bar.progress((i + 1) / (len(uploaded_files) + 2))

    # Call LLM for PDFs that need extraction
    if pdf_texts:
        status_text.text("调用DeepSeek解析PDF...")
        progress_bar.progress(0.6)
        combined = "\n\n---\n\n".join(pdf_texts)
        user_prompt = f"项目名称：{project_name}\n\n文件内容：\n{combined}"
        try:
            llm_raw = call_deepseek(api_key, TRAVEL_EXTRACT_PROMPT, user_prompt, max_tokens=4096)
            llm_data = parse_json_response(llm_raw)
            # Merge LLM results
            for item in llm_data:
                for fd in file_data:
                    if fd["文件名"] == item.get("文件名", "") and fd["类型"] == "pdf":
                        fd["姓名"] = item.get("姓名", "")
                        fd["金额"] = float(item.get("票面金额", 0))
                        fd["保险费"] = float(item.get("保险费", 0))
                        fd["日期"] = item.get("日期", "")
                        fd["文件类型"] = item.get("文件类型", "普通发票")
                        break
        except Exception as e:
            st.warning(f"LLM解析部分失败: {e}，将使用文件名提取姓名")

    # Extract names from filenames for entries without names
    for fd in file_data:
        if not fd.get("姓名"):
            name = Path(fd["文件名"]).stem
            # Try to extract Chinese name (2-4 chars) from filename
            cn_names = re.findall(r'[一-鿿]{2,4}', name)
            if cn_names:
                # Use the last occurrence which is less likely to be a place/company name
                fd["姓名"] = cn_names[-1]
            else:
                fd["姓名"] = "未知"

    progress_bar.progress(0.8, text="匹配老师和金额...")

    # Match teachers with file amounts
    teacher_amounts = {}  # name -> list of amounts
    teacher_insurance = {}  # name -> total insurance
    teacher_details = {}  # name -> list of detail amounts

    for t in teachers:
        name = t["姓名"]
        teacher_amounts[name] = 0.0
        teacher_insurance[name] = 0.0
        teacher_details[name] = []

    # Process each file's amount
    for fd in file_data:
        p_name = fd.get("姓名", "")
        amt = fd.get("金额", 0)
        insurance = fd.get("保险费", 0)
        # In OFD, insurance might be embedded in TotalAmount already
        # For flight tickets with insurance: add insurance to total
        file_type = fd.get("文件类型", "")

        # Try to match name with teachers
        matched = False
        for t in teachers:
            t_name = t["姓名"]
            if p_name and (p_name in t_name or t_name in p_name):
                if file_type == "飞机行程单" and insurance > 0:
                    teacher_amounts[t_name] += amt + insurance
                    teacher_details[t_name].append(amt + insurance)
                else:
                    teacher_amounts[t_name] += amt
                    teacher_details[t_name].append(amt)
                matched = True
                break

        if not matched and p_name:
            # Teacher not in list, add dynamically
            teachers.append({"姓名": p_name, "银行卡号": ""})
            if file_type == "飞机行程单" and insurance > 0:
                teacher_amounts[p_name] = amt + insurance
                teacher_details[p_name] = [amt + insurance]
            else:
                teacher_amounts[p_name] = amt
                teacher_details[p_name] = [amt]

    progress_bar.progress(0.9, text="生成Excel...")

    # Build final data
    output_teachers = []
    total = 0
    for t in teachers:
        name = t["姓名"]
        amt = teacher_amounts.get(name, 0)
        detail = teacher_details.get(name, [])
        output_teachers.append({
            "姓名": f"段 {name}",
            "银行卡号": t.get("银行卡号", ""),
            "实发金额": amt,
            "明细金额": detail
        })
        total += amt

    progress_bar.progress(1.0)
    status_text.text("✅ 完成")

    # Show results
    st.subheader("📊 处理结果")

    # Table
    table_rows = []
    for ot in output_teachers:
        detail_str = " + ".join([f"{v:.2f}" for v in ot["明细金额"]])
        table_rows.append({
            "姓名": ot["姓名"],
            "银行卡号": ot["银行卡号"],
            "实发金额": ot["实发金额"],
            "明细": detail_str
        })
    st.dataframe(table_rows, use_container_width=True, hide_index=True)

    # Total
    st.metric("实发总金额", f"¥{total:.2f}")

    # Generate Excel
    excel_bytes = generate_travel_excel(output_teachers, project_name, project_code)
    st.download_button(
        "📥 下载差旅费Excel",
        data=excel_bytes,
        file_name=f"{project_name}_差旅费.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary"
    )

    # Show raw parsed data (debug)
    with st.expander("查看文件解析详情"):
        for fd in file_data:
            st.json(fd)


def render_tab_labor(project_name, project_code):
    """老师劳务费发放模块"""
    st.subheader("老师劳务费发放")
    st.markdown("无需上传文件，粘贴老师信息即可生成劳务费Excel")

    # Paste teacher info
    st.markdown("**粘贴老师信息** — 姓名、银行卡号、身份证号、手机号")
    info_text = st.text_area(
        "支持表格或文本格式，每行一位老师",
        placeholder="例（tab/空格/逗号分隔）：\n李阳	5月13日上午	讲课	13911080938	222325197012140317	6222600910058226273	1000	4\n刘山	5月13日下午	讲课	18982126401	510124196004260410	4367423818521407216	750	12\n王芳	5月15日全天	讲课	13601023762	110108196512152251	9558800200125094454	1000	8",
        height=200, key="labor_info"
    )

    # Input fields for each row - or parse from pasted text
    if info_text.strip():
        st.markdown("---")
        st.markdown("**解析结果预览**（可修改）")

        # Parse the pasted text
        lines = [l.strip() for l in info_text.strip().split("\n") if l.strip()]
        parsed_teachers = []
        for line in lines:
            parts = re.split(r"[\t,，、\s]{1,}", line)
            parts = [p.strip() for p in parts if p.strip()]
            if len(parts) >= 3:
                teacher = {
                    "姓名": parts[0],
                    "执行时间": parts[1] if len(parts) > 1 else "",
                    "工作内容": parts[2] if len(parts) > 2 else "讲课",
                    "手机号": "",
                    "身份证号": "",
                    "银行卡号": "",
                    "标准": 0,
                    "学时": 0
                }
                for p in parts[3:]:
                    if re.match(r"^\d{11}$", p):
                        teacher["手机号"] = p
                    elif re.match(r"^\d{17}[\dXx]$", p) or re.match(r"^\d{18}$", p):
                        teacher["身份证号"] = p
                    elif re.match(r"^\d{15,}$", p) and len(p) >= 15:
                        teacher["银行卡号"] = p
                    elif re.match(r"^\d+(\.\d+)?$", p):
                        if not teacher["标准"]:
                            teacher["标准"] = float(p)
                        else:
                            teacher["学时"] = float(p)
                parsed_teachers.append(teacher)

        if parsed_teachers:
            st.dataframe(parsed_teachers, use_container_width=True, hide_index=True)
            teachers = parsed_teachers
        else:
            st.warning("无法解析粘贴内容，请检查格式")
            teachers = []
    else:
        teachers = []

    if st.button("🚀 生成劳务费Excel", type="primary", key="labor_btn"):
        if not teachers:
            st.warning("请先粘贴老师信息")
            return

        excel_bytes = generate_labor_excel(teachers, project_name, project_code)
        st.download_button(
            "📥 下载劳务费Excel",
            data=excel_bytes,
            file_name=f"{project_name}_劳务费.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary"
        )

        # Show total
        total = sum(float(t.get("标准", 0)) * float(t.get("学时", 0)) for t in teachers)
        st.metric("总金额", f"¥{total:.2f}")


if __name__ == "__main__":
    main()
