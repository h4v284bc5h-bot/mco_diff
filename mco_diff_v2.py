import streamlit as st
import fitz  # PyMuPDF
import cv2
import numpy as np
import re

# 設定頁面佈局
st.set_page_config(layout="wide", page_title="SPC/FAI 智能自動差異檢測")

# ==========================================
# 1. 影像處理與差異檢測 (新增核心)
# ==========================================

def compare_images_cv2(img1, img2):
    """
    比較兩張圖片，回傳是否不同，以及標示差異後的圖片
    """
    # 1. 確保尺寸一致 (以 img1 為基準)
    h1, w1 = img1.shape[:2]
    img2_resized = cv2.resize(img2, (w1, h1))
    
    # 2. 轉灰階與高斯模糊 (去除雜訊與抗鋸齒誤差)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_RGB2GRAY)
    
    gray1 = cv2.GaussianBlur(gray1, (5, 5), 0)
    gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)
    
    # 3. 計算絕對差異
    diff = cv2.absdiff(gray1, gray2)
    
    # 4. 二值化差異圖 (設定門檻值，濾掉微小誤差)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    
    # 5. 尋找差異輪廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    has_diff = False
    result_img = img2_resized.copy() # 在新圖上標記
    
    for cnt in contours:
        # 忽略太小的噪點面積
        if cv2.contourArea(cnt) > 20:
            has_diff = True
            x, y, w, h = cv2.boundingRect(cnt)
            # 畫出黃色框框標示差異
            cv2.rectangle(result_img, (x, y), (x + w, y + h), (255, 255, 0), 2)
    
    return has_diff, result_img

def run_batch_comparison(doc1, data1, doc2, data2):
    """
    輪詢所有共同的 Key，進行影像比對
    """
    diff_results = {} # 儲存比對結果 {key: True/False}
    
    # 取得共同 Keys
    keys1 = set(data1["FAI"].keys()) | set(data1["SPC"].keys())
    keys2 = set(data2["FAI"].keys()) | set(data2["SPC"].keys())
    common_keys = list(keys1 & keys2)
    
    progress_bar = st.progress(0)
    
    for i, key in enumerate(common_keys):
        # 判斷類別
        cat = "FAI" if "FAI" in key else "SPC"
        
        item1 = data1[cat][key]
        item2 = data2[cat][key]
        
        # 渲染局部圖 (使用較低解析度 2.0x 進行快速比對)
        # 注意：這裡 margin 設小一點，只比對泡泡本體與緊鄰文字
        img1 = render_smart_crop(doc1[item1['page']], item1['rect'], dpi_scale=2.0, margin=10, draw_cross=False)
        img2 = render_smart_crop(doc2[item2['page']], item2['rect'], dpi_scale=2.0, margin=10, draw_cross=False)
        
        is_diff, _ = compare_images_cv2(img1, img2)
        
        if is_diff:
            diff_results[key] = True
            
        progress_bar.progress((i + 1) / len(common_keys))
        
    progress_bar.empty()
    return diff_results

# ==========================================
# 2. 核心解析引擎 (維持向量邏輯)
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
                    label = f"FAI-{num_match.group()}"
                    item["label"] = label
                    item["sort_val"] = int(num_match.group())
                    fai_dict[label] = item
            elif "SPC" in top_str:
                alpha_match = re.search(r'[A-Z]+', bot_str_clean.upper())
                if alpha_match:
                    label = f"SPC-{alpha_match.group()}"
                    item["label"] = label
                    item["sort_val"] = alpha_match.group()
                    spc_dict[label] = item

    return fai_dict, spc_dict

# ==========================================
# 3. 視覺化工具
# ==========================================

def render_smart_crop(page, rect, dpi_scale=4.0, margin=80, draw_cross=True):
    mat = fitz.Matrix(dpi_scale, dpi_scale)
    clip = fitz.Rect(rect.x0 - margin, rect.y0 - margin, rect.x1 + margin, rect.y1 + margin)
    pix = page.get_pixmap(matrix=mat, clip=clip)
    img = cv2.imdecode(np.frombuffer(pix.tobytes(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if draw_cross:
        h, w = img.shape[:2]
        rel_cx = (rect.x0 - clip.x0 + rect.width/2) * dpi_scale
        rel_cy = (rect.y0 - clip.y0 + rect.height/2) * dpi_scale
        cx, cy = int(rel_cx), int(rel_cy)
        r = int((rect.width/2) * dpi_scale)
        
        color = (255, 0, 0)
        thickness = 3 
        
        cv2.line(img, (0, cy), (w, cy), color, 1)
        cv2.line(img, (cx, 0), (cx, h), color, 1)
        cv2.circle(img, (cx, cy), r, color, thickness)
    
    return img

def render_map(page, rect):
    zoom_map = 2.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom_map, zoom_map))
    img = cv2.imdecode(np.frombuffer(pix.tobytes(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    x = int((rect.x0 + rect.x1)/2 * zoom_map)
    y = int((rect.y0 + rect.y1)/2 * zoom_map)
    h, w = img.shape[:2]
    
    line_thickness = 4 
    circle_radius = 50 
    
    cv2.line(img, (0, y), (w, y), (255, 0, 0), 2)
    cv2.line(img, (x, 0), (x, h), (255, 0, 0), 2)
    cv2.circle(img, (x, y), circle_radius, (255, 0, 0), line_thickness)
    
    return img

# ==========================================
# 4. Streamlit UI
# ==========================================

st.title("🛡️ SPC/FAI 智能自動差異檢測")

# Session Init
if 'data_1' not in st.session_state: st.session_state['data_1'] = None
if 'data_2' not in st.session_state: st.session_state['data_2'] = None
if 'bytes_1' not in st.session_state: st.session_state['bytes_1'] = None
if 'bytes_2' not in st.session_state: st.session_state['bytes_2'] = None
if 'diff_results' not in st.session_state: st.session_state['diff_results'] = {}

# --- 側邊欄：檔案上傳 ---
with st.sidebar:
    st.header("1. 檔案載入")
    f1 = st.file_uploader("檔案 1 (基準)", type="pdf", key="f1")
    f2 = st.file_uploader("檔案 2 (對照)", type="pdf", key="f2")

    # 處理檔案 1
    if f1:
        curr_bytes = f1.getvalue()
        if st.session_state['bytes_1'] != curr_bytes:
            st.session_state['bytes_1'] = curr_bytes
            with st.spinner("解析檔案 1..."):
                doc = fitz.open(stream=curr_bytes, filetype="pdf")
                f_d, s_d = analyze_bubbles(doc)
                st.session_state['data_1'] = {"FAI": f_d, "SPC": s_d}
                # Reset diff results
                st.session_state['diff_results'] = {}

    # 處理檔案 2
    if f2:
        curr_bytes = f2.getvalue()
        if st.session_state['bytes_2'] != curr_bytes:
            st.session_state['bytes_2'] = curr_bytes
            with st.spinner("解析檔案 2..."):
                doc = fitz.open(stream=curr_bytes, filetype="pdf")
                f_d, s_d = analyze_bubbles(doc)
                st.session_state['data_2'] = {"FAI": f_d, "SPC": s_d}
                # Reset diff results
                st.session_state['diff_results'] = {}
    else:
        st.session_state['bytes_2'] = None
        st.session_state['data_2'] = None
        st.session_state['diff_results'] = {}

# --- 自動比對邏輯 (Trigger) ---
d1 = st.session_state['data_1']
d2 = st.session_state['data_2']
diff_res = st.session_state['diff_results']

# 當兩個檔案都準備好，且尚未進行比對時，觸發比對
if d1 and d2 and not diff_res:
    with st.spinner("🔄 正在輪詢並比對所有標記差異..."):
        doc1 = fitz.open(stream=st.session_state['bytes_1'], filetype="pdf")
        doc2 = fitz.open(stream=st.session_state['bytes_2'], filetype="pdf")
        
        # 執行批次比對
        results = run_batch_comparison(doc1, d1, doc2, d2)
        st.session_state['diff_results'] = results
        st.success(f"比對完成！發現 {len(results)} 處變更。")

# --- UI 顯示邏輯 ---
if d1:
    with st.sidebar:
        st.divider()
        st.header("2. 標記列表")
        
        cat_mode = st.radio("類別", ["FAI (數字)", "SPC (字母)"], horizontal=True)
        target_key = "FAI" if "FAI" in cat_mode else "SPC"
        
        keys_1 = set(d1[target_key].keys()) if d1 else set()
        keys_2 = set(d2[target_key].keys()) if d2 else set()
        all_keys = list(keys_1 | keys_2)
        
        # 排序
        if target_key == "FAI":
            all_keys.sort(key=lambda x: int(x.split('-')[1]))
        else:
            all_keys.sort(key=lambda x: (len(x.split('-')[1]), x.split('-')[1]))
        
        options = []
        for k in all_keys:
            icon = ""
            if d2: 
                in_1 = k in keys_1
                in_2 = k in keys_2
                
                if in_1 and in_2:
                    # 檢查是否有內容差異
                    if k in st.session_state['diff_results']:
                        icon = "⚠️ " # 差異!
                    else:
                        icon = "✅ " # 無差異
                elif in_1 and not in_2: icon = "❌ "
                elif not in_1 and in_2: icon = "🆕 "
            else:
                icon = "📍 "
            
            options.append(f"{icon}{k}")
            
        if not options:
            st.warning("無此類別資料")
            sel_key = None
        else:
            sel_opt = st.radio("選擇標記:", options, label_visibility="collapsed")
            sel_key = sel_opt.split(" ")[1] if " " in sel_opt else sel_opt

    # --- 主畫面 ---
    if sel_key:
        # 判斷是否為「差異」項目
        is_modified = sel_key in st.session_state['diff_results']
        status_text = " (⚠️ 偵測到變更)" if is_modified else ""
        
        st.subheader(f"{sel_opt} 同步檢視 {status_text}")
        
        view_scope = st.slider(
            "視野範圍 (Field of View)", 
            min_value=50, max_value=300, 
            value=100, step=10
        )
        
        c1, c2 = st.columns(2)
        
        # --- File 1 Render ---
        with c1:
            st.markdown("### 📄 檔案 1")
            if d1 and sel_key in d1[target_key]:
                item = d1[target_key][sel_key]
                doc1 = fitz.open(stream=st.session_state['bytes_1'], filetype="pdf")
                page1 = doc1[item['page']]
                
                sub_c1, sub_c2, sub_c3 = st.columns([1.5, 7, 1.5])
                with sub_c2:
                    img_hi = render_smart_crop(page1, item['rect'], dpi_scale=4.0, margin=view_scope)
                    st.image(img_hi, use_container_width=True)
                
                img_map = render_map(page1, item['rect'])
                st.image(img_map, use_container_width=True)
            else:
                st.warning("無此標記")
                
        # --- File 2 Render ---
        with c2:
            st.markdown("### 📄 檔案 2")
            if d2 and sel_key in d2[target_key]:
                item = d2[target_key][sel_key]
                doc2 = fitz.open(stream=st.session_state['bytes_2'], filetype="pdf")
                page2 = doc2[item['page']]
                
                sub_c1, sub_c2, sub_c3 = st.columns([1.5, 7, 1.5])
                with sub_c2:
                    # 渲染基礎圖
                    img_hi = render_smart_crop(page2, item['rect'], dpi_scale=4.0, margin=view_scope)
                    
                    # [關鍵功能 3] 如果有差異，在 Local Zoom 畫面上畫出差異框
                    if is_modified and d1 and sel_key in d1[target_key]:
                        # 為了畫出差異，我們需要再拿 File 1 的圖來比對一次 (這次是用目前的高解析度設定)
                        item1 = d1[target_key][sel_key]
                        page1 = doc1[item1['page']]
                        img1_for_diff = render_smart_crop(page1, item1['rect'], dpi_scale=4.0, margin=view_scope, draw_cross=False)
                        
                        # 產生沒有十字線的 File 2 圖來做乾淨比對
                        img2_clean = render_smart_crop(page2, item['rect'], dpi_scale=4.0, margin=view_scope, draw_cross=False)
                        
                        # 計算差異並畫在 img_hi (有十字線的圖) 上
                        # 我們呼叫 compare_images_cv2，但我們要把它畫在 img_hi 上
                        _, diff_overlay = compare_images_cv2(img1_for_diff, img_hi) # 注意: 這裡傳入 img_hi 讓框框畫在有十字的圖上
                        st.image(diff_overlay, caption="⚠️ 差異標示 (黃框)", use_container_width=True)
                    else:
                        st.image(img_hi, use_container_width=True)
                    
                img_map = render_map(page2, item['rect'])
                st.image(img_map, use_container_width=True)
            else:
                st.warning("無此標記")
else:
    st.info("請先上傳檔案 1。")
