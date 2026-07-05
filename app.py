import streamlit as st
import os
import glob
import nest_asyncio
from PIL import Image
import tempfile
import shutil
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, PromptTemplate, Settings, Document
import google.generativeai as genai
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_parse import LlamaParse

# 確保非同步設定
nest_asyncio.apply()

# --- 介面設定 ---
st.set_page_config(page_title="PCM 職安衛審查系統", layout="wide")
st.title("🛡️ PCM 職安衛 / 契約文件審查系統")

# ==========================================
# 自動讀取 secrets.toml
# ==========================================
st.sidebar.header("API 設定")

try:
    default_google_key = st.secrets["GOOGLE_API_KEY"]
    default_llama_key = st.secrets["LLAMAPARSE_KEY"]
except Exception:
    default_google_key = ""
    default_llama_key = ""

google_key = st.sidebar.text_input("Google API Key", value=default_google_key, type="password")
llm_parse_key = st.sidebar.text_input("LlamaParse Key", value=default_llama_key, type="password")

# --- 邏輯初始化 ---
def init_models(g_key, l_key):
    os.environ["GOOGLE_API_KEY"] = g_key
    genai.configure(api_key=g_key)
    Settings.llm = GoogleGenAI(model="models/gemini-3.5-flash", temperature=0.0, api_key=g_key)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")
    return genai.GenerativeModel('gemini-3.5-flash'), LlamaParse(api_key=l_key, result_type="markdown", language="ch_tra")

# --- 檔案上傳區 ---
st.header("1. 上傳資料")
col1, col2 = st.columns(2)
std_files = col1.file_uploader("上傳法規/規範契約檔案 (osh_standards)", accept_multiple_files=True)
rep_files = col2.file_uploader("上傳廠商送審文件 (PDF/圖片)", accept_multiple_files=True)

# ==========================================
# 🌟 新增功能：讓使用者自由輸入審查項目
# ==========================================
st.header("2. 設定審查條件")
inspection_item = st.text_input(
    "請輸入您想查核的項目或條件：", 
    value="人員相關證照是否過期", 
    help="例如：人員相關證照是否過期、施工架組裝標準、開挖擋土設施規定等。"
)

st.markdown("---")

if st.button("🚀 開始審查"):
    if not google_key or not llm_parse_key:
        st.error("請確認側邊欄已輸入 API Keys")
    elif not std_files or not rep_files:
        st.warning("⚠️ 請確保兩邊（法規標準與廠商文件）都有上傳檔案！")
    elif not inspection_item.strip():
        st.warning("⚠️ 請輸入您想查核的項目！")
    else:
        vision_model, parser = init_models(google_key, llm_parse_key)
        
        # 建立暫存資料夾
        temp_dir = tempfile.mkdtemp()
        std_dir = os.path.join(temp_dir, "osh")
        rep_dir = os.path.join(temp_dir, "rep")
        os.makedirs(std_dir); os.makedirs(rep_dir)

        # 儲存上傳的檔案到暫存區
        for f in std_files:
            with open(os.path.join(std_dir, f.name), "wb") as w: w.write(f.getbuffer())
        
        rep_image_paths = []
        for f in rep_files:
            path = os.path.join(rep_dir, f.name)
            with open(path, "wb") as w: w.write(f.getbuffer())
            if f.name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                rep_image_paths.append(path)

        # --- 開始審查 ---
        with st.spinner(f"🧠 正在依據「{inspection_item}」讀取大腦並分析中，請稍候..."):
            try:
                # 讀取法規
                std_docs = SimpleDirectoryReader(std_dir, file_extractor={".pdf": parser}).load_data()
                std_engine = VectorStoreIndex.from_documents(std_docs).as_query_engine(similarity_top_k=5)
                
                # 讀取廠商文件 (排除單獨圖片)
                rep_docs = []
                try:
                    rep_docs.extend(SimpleDirectoryReader(rep_dir, file_extractor={".pdf": parser}).load_data())
                except ValueError:
                    pass
                
                # 處理單獨圖片 (將使用者輸入的條件帶入圖片解析的提示詞中)
                for img_path in rep_image_paths:
                    img = Image.open(img_path)
                    prompt = f"請詳細辨識這張圖片中的所有文字內容。我們目前的 PCM 審查重點為：「{inspection_item}」。請盡可能萃取出與此查核項目相關的關鍵資訊（例如：姓名、日期、數值、尺寸、發證單位等），若無直接相關也請提供清晰的文字轉錄。"
                    response = vision_model.generate_content([prompt, img])
                    rep_docs.append(Document(text=f"【影像內容：{os.path.basename(img_path)}】\n{response.text}", metadata={"file": os.path.basename(img_path)}))
                
                if not rep_docs:
                    st.error("❌ 找不到可解析的廠商文件或圖片！")
                else:
                    rep_engine = VectorStoreIndex.from_documents(rep_docs).as_query_engine(similarity_top_k=5)
                    
                    # 審查比對 (讓法規大腦依據自訂條件搜尋)
                    found_laws = std_engine.query(f"請找出關於『{inspection_item}』的法規或契約規範依據。")
                    
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

        # 清除暫存
        shutil.rmtree(temp_dir)
