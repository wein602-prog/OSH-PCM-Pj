import asyncio
import nest_asyncio
nest_asyncio.apply()
# 如果依然報錯，可以強制設定一個新的迴圈：
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())
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
st.title("🛡️ PCM 職安衛文件審查系統")

# --- 側邊欄 API 設定 (從 secrets 讀取) ---
st.sidebar.header("API 設定")
# 若未設定 secrets，則顯示輸入框
google_key = st.sidebar.text_input("Google API Key", type="password")
llm_parse_key = st.sidebar.text_input("LlamaParse Key", type="password")

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
std_files = col1.file_uploader("上傳法規檔案 (osh_standards)", accept_multiple_files=True)
rep_files = col2.file_uploader("上傳廠商審查文件 (PDF/圖片)", accept_multiple_files=True)

if st.button("🚀 開始審查"):
    if not google_key or not llm_parse_key:
        st.error("請在側邊欄輸入 API Keys")
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
        with st.spinner("🧠 正在讀取大腦並分析中..."):
            # 讀取法規
            std_docs = SimpleDirectoryReader(std_dir, file_extractor={".pdf": parser}).load_data()
            std_engine = VectorStoreIndex.from_documents(std_docs).as_query_engine(similarity_top_k=5)
            
            # 讀取廠商文件 (排除圖片，圖片我們另外用 Vision 模型處理)
            rep_docs = SimpleDirectoryReader(rep_dir, file_extractor={".pdf": parser}).load_data()
            
            # 處理單獨圖片
            for img_path in rep_image_paths:
                img = Image.open(img_path)
                prompt = "請辨識圖片文字，如果是證照請列出姓名、有效期限等資訊。"
                response = vision_model.generate_content([prompt, img])
                rep_docs.append(Document(text=f"【影像內容】{response.text}", metadata={"file": os.path.basename(img_path)}))
            
            rep_engine = VectorStoreIndex.from_documents(rep_docs).as_query_engine(similarity_top_k=5)
            
            # 審查
            found_laws = std_engine.query("請找出關於『人員相關證照是否過期』的法規依據。")
            final_report = rep_engine.query(str(found_laws))
            
            st.success("✅ 審查完成")
            st.markdown("### 📄 審查報告")
            st.write(final_report.response)

        # 清除暫存
        shutil.rmtree(temp_dir)
