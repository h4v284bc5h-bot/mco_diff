import streamlit as st
import fitz  # PyMuPDF
import cv2
import numpy as np
import re

# ==========================================
# 0. 初始化與設定
# ==========================================
st.set_page_config(layout="wide", page_title="Display MPE MCO智能比較系統")

# [修改] CSS 樣式表：
# 1. 縮小側邊欄按鈕字體 (原本需求)
# 2. 強制讓按鈕填滿寬度 (取代 use_container_width=True)
# 3. 確保圖片在欄位中自適應
st.markdown("""
<style>
    /* 側邊欄按鈕樣式優化 */
    div[data-testid="stSidebar"] div.stButton > button {
        width: 100%;                     /* 強制填滿寬度 */
        font-size: 12px !important;      /* 字體縮小 */
        padding-top: 4px !important;     /* 減少上方留白 */
        padding-bottom: 4px !important;  /* 減少下方留白 */
        min-height: 0px !important;      /* 移除最小高度限制 */
        height: auto !important;         /* 高度自動 */
        line-height: 1.2 !important;     /* 行高緊湊 */
    }
    
    /* 確保圖片填滿容器 */
    div[data-testid="stImage"] > img {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# 初始化 Session State
if 'data_1' not in st.session_state: st.session_state['data_1'] = None
if 'data_2' not in st.session_state: st.session_state['data_2'] = None
if 'bytes_1' not in st.session_state: st.session_state['bytes_1'] = None
if 'bytes_2' not in st.session_state: st.session_state['bytes_2'] = None
if 'name_1' not in st.session_state: st.session_state['name_1'] = None
if 'name_2' not in st.session_state: st.session_state['name_2'] = None
if 'diff_results' not in st.session_state: st.session_state['diff_results'] = {}

def reset_diff_state():
    st.session_state['diff_results'] = {}

# ==========================================
# 1. 頂部佈局 (標題 + 視野拉桿)
# ==========================================

c_header, c_slider = st.columns([7, 3], vertical_alignment="bottom")

with c_header:
    st.title("🛡️ Display MPE MCO智能比較系統")

with c_slider:
    view_scope = st.slider(
        "視野範圍 (Field of View)", 
        min_value=50, max_value=300, 
        value=120, step=10, 
        help="數值越小=特寫越近，數值越大=看到越多周圍環境",
        on_change=reset_diff_state
    )

# ==========================================
# 2. 影像處理與差異檢測
# ==========================================

def pixmap_to_cv2(pix):
    if pix.colorspace.n != 3:
        pix = fitz.Pixmap(fitz.csRGB, pix)
    img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n).copy()
    return img_array

@st.cache_data(show_spinner=False)
def get_cached_base_map(file_bytes, page_num, zoom=2.0):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    page = doc[page_num]
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    return pixmap_to_cv2(pix)

def render_map_fast(file_bytes, page_num, rect, zoom_map=2.0):
    base_img = get_cached_base_map(file_bytes, page_num, zoom_map)
    img = base_img.copy()
    x = int((rect.x0 + rect.x1)/2 * zoom_map)
    y = int((rect.y0 + rect.y1)/2 * zoom_map)
    h, w = img.shape[:2]
    cv2.line(img, (0, y), (w, y), (255, 0, 0), 2)
    cv2.line(img, (x, 0), (x, h), (255, 0, 0), 2)
    cv2.circle(img, (x, y), 50, (255, 0, 0), 4)
    return img

def render_smart_crop_fast(page, rect, dpi_scale=3.0, margin=120, draw_cross=True):
    mat = fitz.Matrix(dpi_scale, dpi_scale)
    clip_request = fitz.Rect(rect.x0 - margin, rect.y0 - margin, rect.x1 + margin, rect.y1 + margin)
    final_clip = clip_request & page.rect
    pix = page.get_pixmap(matrix=mat, clip=final_clip)
    img = pixmap_to_cv2(pix)
    
    if draw_cross:
        h, w = img.shape[:2]
        center_x = (rect.x0 + rect.x1) / 2
        center_y = (rect.y0 + rect.y1) / 2
        rel_cx = (center_x - final_clip.x0) * dpi_scale
        rel_cy = (center_y - final_clip.y0) * dpi_scale
        cx, cy = int(rel_cx), int(rel_cy)
        r = int((rect.width/2) * dpi_scale)
        color = (255, 0, 0)
        thickness = max(2, int(dpi_scale * 0.8))
        try:
            cv2.line(img, (0, cy), (w, cy), color, 1)
            cv2.line(img, (cx, 0), (cx, h), color, 1)
            cv2.circle(img, (cx, cy), r, color, thickness)
        except: pass
    return img

def compare_images_cv2(img1, img2):
    h1, w1 = img1.shape[:2]
    img2_resized = cv2.resize(img2, (w1, h1))
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_RGB2GRAY)
    gray1 = cv2.GaussianBlur(gray1, (5, 5), 0)
    gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)
    diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    has_diff = False
    result_img = img2_resized.copy()
    for cnt in contours:
        if cv2.contourArea(cnt) > 20:
            has_diff = True
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(result_img, (x, y), (x + w, y + h), (255, 255, 0), 2)
    return has_diff, result_img

def run_batch_comparison(doc1, data1, doc2, data2, margin_val):
    diff_results = {}
    keys1 = set(data1["FAI"].keys()) | set(data1["SPC"].keys())
    keys2 = set(data2["FAI"].keys()) | set(data2["SPC"].keys())
    common_keys = list(keys1 & keys2)
    progress_bar = st.progress(0)
    for i, key in enumerate(common_keys):
        cat = "FAI" if "FAI" in key else "SPC"
        item1 = data1[cat][key]
        item2 = data2[cat][key]
        img1 = render_smart_crop_fast(doc1[item1['page']], item1['rect'], dpi_scale=2.0, margin=margin_val, draw_cross=False)
        img2 = render_smart_crop_fast(doc2[item2['page']], item2['rect'], dpi_scale=2.0, margin=margin_val, draw_cross=False)
        is_diff, _ = compare_images_cv2(img1, img2)
        if is_diff:
            diff_results[key] = True
        progress_bar.progress((i + 1) / len(common_keys))
    progress_bar.empty()
    return diff_results

# ==========================================
# 3. 核心解析引擎
# ==========================================

def get_text_spans(page):
    spans = []
    text_dict = page.get_text("dict")
    for block in text_dict["blocks"]:
        if block["type"] == 0:
            for line in block["lines"]:
                for span in line["spans"]:
                    bbox = fitz.Rect(span["bbox"])
                    center = ((bbox.x0 + bbox.x1) / 2, (bbox.y0 + bbox.y1) / 2)
                    spans.append({"text": span["text"].strip(), "center": center})
    return spans

def is_vector_circle(path):
    rect = path["rect"]
    if not (5 < rect.width < 300 and 5 < rect.height < 300): return False
    if rect.height == 0: return False
    if not (0.85 <= rect.width / rect.height <= 1.15): return False
    has_curve = any(item[0] == 'c' for item in path["items"])
    return has_curve

def analyze_bubbles(doc):
    fai_dict = {}
    spc_dict = {}
    for page_num, page in enumerate(doc):
        text_spans = get_text_spans(page)
        paths = page.get_drawings()
        for path in paths:
            if not is_vector_circle(path): continue
            rect = path["rect"]
            cx, cy = (rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2
            top_texts = [s["text"] for s in text_spans if rect.contains(s["center"]) and s["center"][1] < cy]
            bot_texts = [s["text"] for s in text_spans if rect.contains(s["center"]) and s["center"][1] > cy]
            top_str = "".join(top_texts).upper()
            bot_str = "".join(bot_texts).strip()
            bot_str_clean = re.sub(r'[-_]', '', bot_str)
            item = {"page": page_num, "rect": rect}
            if "FAI" in top_str:
                num_match = re.search(r'\d+', bot_str_clean)
                if num_match:
                    item["label"] = f"FAI-{num_match.group()}"
                    item["sort_val"] = int(num_match.group())
                    fai_dict[item["label"]] = item
            elif "SPC" in top_str:
                alpha_match = re.search(r'[A-Z]+', bot_str_clean.upper())
                if alpha_match:
                    item["label"] = f"SPC-{alpha_match.group()}"
                    item["sort_val"] = alpha_match.group()
                    spc_dict[item["label"]] = item
    return fai_dict, spc_dict

# ==========================================
# 4. Streamlit UI (側邊欄)
# ==========================================

with st.sidebar:
    st.header("1. 檔案載入")
    f1 = st.file_uploader("檔案 1 (基準)", type="pdf", key="f1")
    f2 = st.file_uploader("檔案 2 (對照)", type="pdf", key="f2")

    if f1:
        st.session_state['name_1'] = f1.name
        curr_bytes = f1.getvalue()
        if st.session_state['bytes_1'] != curr_bytes:
            st.session_state['bytes_1'] = curr_bytes
            with st.spinner("解析檔案 1..."):
                doc = fitz.open(stream=curr_bytes, filetype="pdf")
                f_d, s_d = analyze_bubbles(doc)
                st.session_state['data_1'] = {"FAI": f_d, "SPC": s_d}
                st.session_state['diff_results'] = {}
    else:
        st.session_state['data_1'] = None
        st.session_state['bytes_1'] = None
        st.session_state['name_1'] = None
        st.session_state['diff_results'] = {}

    if f2:
        st.session_state['name_2'] = f2.name
        curr_bytes = f2.getvalue()
        if st.session_state['bytes_2'] != curr_bytes:
            st.session_state['bytes_2'] = curr_bytes
            with st.spinner("解析檔案 2..."):
                doc = fitz.open(stream=curr_bytes, filetype="pdf")
                f_d, s_d = analyze_bubbles(doc)
                st.session_state['data_2'] = {"FAI": f_d, "SPC": s_d}
                st.session_state['diff_results'] = {}
    else:
        st.session_state['data_2'] = None
        st.session_state['bytes_2'] = None
        st.session_state['name_2'] = None
        st.session_state['diff_results'] = {}

d1 = st.session_state['data_1']
d2 = st.session_state['data_2']
diff_res = st.session_state['diff_results']

# 自動觸發比對
if d1 and d2 and not diff_res:
    with st.spinner("🔄 正在比對所有標記差異..."):
        doc1 = fitz.open(stream=st.session_state['bytes_1'], filetype="pdf")
        doc2 = fitz.open(stream=st.session_state['bytes_2'], filetype="pdf")
        results = run_batch_comparison(doc1, d1, doc2, d2, margin_val=view_scope)
        st.session_state['diff_results'] = results
        st.success(f"比對完成！發現 {len(results)} 處變更。")

if d1 or d2:
    with st.sidebar:
        st.divider()
        st.header("2. 標記列表")
        
        # 預設 SPC
        cat_mode = st.radio("類別", ["SPC (字母)", "FAI (數字)"], horizontal=True)
        target_key = "SPC" if "SPC" in cat_mode else "FAI"
        
        keys_1 = set(d1[target_key].keys()) if d1 else set()
        keys_2 = set(d2[target_key].keys()) if d2 else set()
        all_keys = list(keys_1 | keys_2)
        
        if target_key == "FAI":
            all_keys.sort(key=lambda x: int(x.split('-')[1]))
        else:
            all_keys.sort(key=lambda x: (len(x.split('-')[1]), x.split('-')[1]))
        
        options = []
        diff_indices = []
        
        for idx, k in enumerate(all_keys):
            icon = ""
            is_diff = False
            
            if d1 and d2:
                in_1 = k in keys_1
                in_2 = k in keys_2
                
                if in_1 and in_2:
                    if k in st.session_state['diff_results']:
                        icon = "⚠️ "
                        is_diff = True
                    else:
                        icon = "✅ "
                elif in_1 and not in_2:
                    icon = "❌ "
                    is_diff = True
                elif not in_1 and in_2:
                    icon = "🆕 "
                    is_diff = True
            else:
                icon = "📍 "
            
            if is_diff:
                diff_indices.append(idx)
                
            options.append(f"{icon}{k}")

        def go_prev_diff():
            current_opt = st.session_state.get('nav_radio')
            if current_opt in options:
                curr_idx = options.index(current_opt)
                prev_candidates = [i for i in diff_indices if i < curr_idx]
                target_idx = prev_candidates[-1] if prev_candidates else (diff_indices[-1] if diff_indices else curr_idx)
                st.session_state['nav_radio'] = options[target_idx]

        def go_next_diff():
            current_opt = st.session_state.get('nav_radio')
            if current_opt in options:
                curr_idx = options.index(current_opt)
                next_candidates = [i for i in diff_indices if i > curr_idx]
                target_idx = next_candidates[0] if next_candidates else (diff_indices[0] if diff_indices else curr_idx)
                st.session_state['nav_radio'] = options[target_idx]

        if d1 and d2 and diff_indices:
            col_b1, col_b2 = st.columns(2)
            # [修改] 移除 use_container_width 以消除警告，依賴 CSS 進行寬度填充
            with col_b1:
                st.button("⬆️ 上一個差異", on_click=go_prev_diff)
            with col_b2:
                st.button("⬇️ 下一個差異", on_click=go_next_diff)
            
            st.caption(f"發現 {len(diff_indices)} 個差異點。")

        if not options:
            st.warning("無此類別資料")
            sel_key = None
        else:
            sel_opt = st.radio("選擇標記:", options, label_visibility="collapsed", key="nav_radio")
            sel_key = sel_opt.split(" ")[1] if " " in sel_opt else sel_opt

    # --- 主畫面 ---
    if sel_key:
        is_modified = sel_key in st.session_state['diff_results']
        status_text = " (⚠️ 變更)" if is_modified else ""
        
        st.subheader(f"{sel_opt} 檢視 {status_text}")
        
        if d1 and d2:
            c1, c2 = st.columns(2)
        elif d1:
            _, c1, _ = st.columns([1, 2, 1])
            c2 = None
        elif d2:
            _, c2, _ = st.columns([1, 2, 1])
            c1 = None
        
        # --- File 1 Render ---
        if c1 and d1:
            with c1:
                name1 = st.session_state['name_1']
                role1 = "(基準)" if d2 else ""
                st.markdown(f"<h4 style='text-align:center;'>📄 {name1}<br><span style='font-size:0.7em; color:gray;'>{role1}</span></h4>", unsafe_allow_html=True)
                
                if sel_key in d1[target_key]:
                    item = d1[target_key][sel_key]
                    img_hi = render_smart_crop_fast(
                        fitz.open(stream=st.session_state['bytes_1'], filetype="pdf")[item['page']], 
                        item['rect'], 
                        dpi_scale=3.0, 
                        margin=view_scope
                    )
                    # [修改] 移除 use_container_width
                    st.image(img_hi)
                    
                    img_map = render_map_fast(st.session_state['bytes_1'], item['page'], item['rect'])
                    st.image(img_map)
                else:
                    st.warning("無此標記")
        
        # --- File 2 Render ---
        if c2 and d2:
            with c2:
                name2 = st.session_state['name_2']
                role2 = "(對照)" if d1 else ""
                st.markdown(f"<h4 style='text-align:center;'>📄 {name2}<br><span style='font-size:0.7em; color:gray;'>{role2}</span></h4>", unsafe_allow_html=True)
                
                if sel_key in d2[target_key]:
                    item = d2[target_key][sel_key]
                    img_hi = render_smart_crop_fast(
                        fitz.open(stream=st.session_state['bytes_2'], filetype="pdf")[item['page']],
                        item['rect'], 
                        dpi_scale=3.0, 
                        margin=view_scope
                    )
                    
                    if d1 and is_modified and sel_key in d1[target_key]:
                        item1 = d1[target_key][sel_key]
                        img1_for_diff = render_smart_crop_fast(
                            fitz.open(stream=st.session_state['bytes_1'], filetype="pdf")[item1['page']],
                            item1['rect'], 
                            dpi_scale=3.0, 
                            margin=view_scope, 
                            draw_cross=False
                        )
                        img2_clean = render_smart_crop_fast(
                            fitz.open(stream=st.session_state['bytes_2'], filetype="pdf")[item['page']],
                            item['rect'], 
                            dpi_scale=3.0, 
                            margin=view_scope, 
                            draw_cross=False
                        )
                        _, diff_overlay = compare_images_cv2(img1_for_diff, img_hi)
                        st.image(diff_overlay, caption="⚠️ 差異標示")
                    else:
                        st.image(img_hi)
                    
                    img_map = render_map_fast(st.session_state['bytes_2'], item['page'], item['rect'])
                    st.image(img_map)
                else:
                    st.warning("無此標記")
else:
    st.info("請上傳至少一個 PDF 檔案以開始。")
