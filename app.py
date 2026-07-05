# ==========================================
# 0. 解決 Streamlit 與 LlamaIndex 的非同步衝突
# (這段必須放在最頂端)
# ==========================================
import asyncio
import nest_asyncio

# 強制獲取或建立一個新的事件迴圈，確保後續 AI 套件不會報錯
try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

nest_asyncio.apply()

# ==========================================
# 1. 匯入必要套件
# ==========================================
import streamlit as st
import os
import glob
from PIL import Image
import tempfile
import shutil
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, PromptTemplate, Settings, Document
import google.generativeai as genai
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_parse import LlamaParse

# ==========================================
# 2. 介面與 API 初始化設定
# ==========================================
st.set_page_config(page_title="PCM 職安衛審查系統", layout="wide")
st.title("🛡️ PCM 職安衛 / 契約文件審查系統")

st.sidebar.header("API 設定")

# 嘗試自動從 Streamlit Secrets 讀取金鑰 (若沒有則預設為空字串)
try:
    default_google_key = st.secrets.get("GOOGLE_API_KEY", "")
    default_llama_key = st.secrets.get("LLAMAPARSE_KEY", "")
except Exception:
    default_google_key = ""
    default_llama_key = ""

google_key = st.sidebar.text_input("Google API Key", value=default_google_key, type="password")
llm_parse_key = st.sidebar.text_input("LlamaParse Key", value=default_llama_key, type="password")

# --- 載入 AI 模型 ---
def init_models(g_key, l_key):
    os.environ["GOOGLE_API_KEY"] = g_key
    genai.configure(api_key=g_key)
    
    # 初始化語言模型與嵌入模型
    Settings.llm = GoogleGenAI(model="models/gemini-3.5-flash", temperature=0.0, api_key=g_key)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")
    
    # 初始化視覺辨識模型與文件解析器
    vision_model = genai.GenerativeModel('gemini-3.5-flash')
    parser = LlamaParse(api_key=l_key, result_type="markdown", language="ch_tra")
    
    return vision_model, parser

# ==========================================
# 3. 使用者操作區 (上傳與設定)
# ==========================================
st.header("1. 上傳資料")
col1, col2 = st.columns(2)
std_files = col1.file_uploader("上傳法規/規範契約檔案 (osh_standards)", accept_multiple_files=True)
rep_files = col2.file_uploader("上傳廠商送審文件 (PDF/圖片)", accept_multiple_files=True)

st.header("2. 設定審查條件")
inspection_item = st.text_input(
    "請輸入您想查核的項目或條件：", 
    value="人員相關證照是否過期", 
    help="例如：人員相關證照是否過期、施工架組裝標準、開挖擋土設施規定等。"
)

st.markdown("---")

# ==========================================
# 4. 核心審查邏輯
# ==========================================
if st.button("🚀 開始審查"):
    # 檢查必填欄位
    if not google_key or not llm_parse_key:
        st.error("請確認側邊欄已輸入 API Keys")
    elif not std_files or not rep_files:
        st.warning("⚠️ 請確保兩邊（法規標準與廠商文件）都有上傳檔案！")
    elif not inspection_item.strip():
        st.warning("⚠️ 請輸入您想查核的項目！")
    else:
        # 初始化模型
        vision_model, parser = init_models(google_key, llm_parse_key)
        
        # 建立暫存資料夾來存放使用者上傳的檔案
        temp_dir = tempfile.mkdtemp()
        std_dir = os.path.join(temp_dir, "osh")
        rep_dir = os.path.join(temp_dir, "rep")
        os.makedirs(std_dir, exist_ok=True)
        os.makedirs(rep_dir, exist_ok=True)

        # 儲存上傳的法規檔案
        for f in std_files:
            with open(os.path.join(std_dir, f.name), "wb") as w: 
                w.write(f.getbuffer())
        
        # 儲存上傳的廠商文件，並單獨過濾出圖片檔
        rep_image_paths = []
        for f in rep_files:
            path = os.path.join(rep_dir, f.name)
            with open(path, "wb") as w: 
                w.write(f.getbuffer())
            if f.name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                rep_image_paths.append(path)

        # --- 開始進行 AI 解析 ---
        with st.spinner(f"🧠 正在依據「{inspection_item}」讀取大腦並分析中，請稍候..."):
            try:
                # 建立法規大腦 (使用 LlamaParse 解析 PDF)
                std_docs = SimpleDirectoryReader(std_dir, file_extractor={".pdf": parser}).load_data()
                std_engine = VectorStoreIndex.from_documents(std_docs).as_query_engine(similarity_top_k=5)
                
                # 建立廠商文件大腦
                rep_docs = []
                try:
                    # 解析 PDF 等一般文件
                    rep_docs.extend(SimpleDirectoryReader(rep_dir, file_extractor={".pdf": parser}).load_data())
                except ValueError:
                    # 若資料夾內只有圖片沒有文件，攔截報錯
                    pass
                
                # 處理單獨圖片檔 (將自訂條件帶入 Vision 模型進行重點萃取)
                for img_path in rep_image_paths:
                    try:
                        img = Image.open(img_path)
                        prompt = f"請詳細辨識這張圖片中的所有文字內容。我們目前的審查重點為：「{inspection_item}」。請盡可能萃取出與此查核項目相關的關鍵資訊，若無直接相關也請提供清晰的文字轉錄。"
                        response = vision_model.generate_content([prompt, img])
                        
                        rep_docs.append(Document(
                            text=f"【影像內容：{os.path.basename(img_path)}】\n{response.text}", 
                            metadata={"file": os.path.basename(img_path)}
                        ))
                    except Exception as img_e:
                        st.warning(f"解析圖片 {os.path.basename(img_path)} 時發生錯誤: {img_e}")
                
                # 確認是否有成功載入任何待審查資料
                if not rep_docs:
                    st.error("❌ 找不到可解析的廠商文件或圖片！")
                else:
                    rep_engine = VectorStoreIndex.from_documents(rep_docs).as_query_engine(similarity_top_k=5)
                    
                    # 進行法規查詢
                    found_laws = std_engine.query(f"請找出關於『{inspection_item}』的法規或契約規範依據。")
                    
                    # 組合最終審查 Prompt
                    pcm_osh_template = f"""你是一位專業 PCM (專案管理) 職安衛主管。請依據以下法規標準審查廠商文件。
                    本次重點查核項目為：【{inspection_item}】
                    
                    【法規標準】{{query_str}}
                    【廠商內容】{{context_str}}
                    
                    請給予：
                    1. 審查項目
                    2. 廠商敘述與數據
                    3. 審查判定 (合格 / 不合格 / 需澄清)
                    4. PCM 具體建議。"""
                    
                    rep_engine.update_prompts({"response_synthesizer:text_qa_template": PromptTemplate(pcm_osh_template)})
                    final_report = rep_engine.query(str(found_laws))
                    
                    st.success("✅ 審查完成！")
                    st.markdown(f"### 📄 審查報告：{inspection_item}")
                    st.write(final_report.response)

            except Exception as e:
                st.error(f"系統執行時發生錯誤：{e}")

        # 執行完畢，清理暫存檔案
        shutil.rmtree(temp_dir, ignore_errors=True)
