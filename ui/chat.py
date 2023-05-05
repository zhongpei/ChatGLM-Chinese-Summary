import gradio as gr
import os
import shutil
from loguru import logger
from utils.chatpdf import ChatPDF
import hashlib
from utils.llm import LLM
from models import MAX_INPUT_LEN, models

pwd_path = os.path.abspath(os.path.dirname(__file__))

CONTENT_DIR = os.path.join(pwd_path, "content")
logger.info(f"CONTENT_DIR: {CONTENT_DIR}")
VECTOR_SEARCH_TOP_K = 3


def get_file_list():
    if not os.path.exists("content"):
        return []
    return [f for f in os.listdir("content") if
            f.endswith(".txt") or f.endswith(".pdf") or f.endswith(".docx") or f.endswith(".md")]


def upload_file(file, file_list):
    if not os.path.exists(CONTENT_DIR):
        os.mkdir(CONTENT_DIR)
    filename = os.path.basename(file.name)
    shutil.move(file.name, os.path.join(CONTENT_DIR, filename))
    # file_list首位插入新上传的文件
    file_list.insert(0, filename)
    return gr.Dropdown.update(choices=file_list, value=filename), file_list


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text


def get_answer(
        query,
        index_path,
        history,
        topn: int = VECTOR_SEARCH_TOP_K,
        max_input_size: int = 1024,
        chat_mode: str = "pdf"
):
    if not models.is_active():
        return [None, "模型还未加载"], query
    if index_path and chat_mode == "pdf":
        if not models.chatpdf.sim_model.corpus_embeddings:
            models.chatpdf.load_index(index_path)
        response, empty_history, reference_results = models.chatpdf.query(
            llm_model=models.llm_model,
            query=query,
            topn=topn,
            max_input_size=max_input_size
        )

        logger.debug(f"query: {query}, response with content: {response}")
        for i in range(len(reference_results)):
            r = reference_results[i]
            response += f"\n{r.strip()}"
        response = parse_text(response)
        history = history + [[query, response]]
    else:
        # 未加载文件，仅返回生成模型结果
        response, empty_history = models.llm_model.chat(query, history)
        response = parse_text(response)
        history = history + [[query, response]]
        logger.debug(f"query: {query}, response: {response}")
    return history, ""


def update_status(history, status):
    history = history + [[None, status]]
    logger.info(status)
    return history


def get_file_hash(fpath):
    return hashlib.md5(open(fpath, 'rb').read()).hexdigest()


def get_vector_store(filepath, history, embedding_model):
    logger.info(filepath, history)
    index_path = None
    file_status = ''
    if models.chatpdf is not None:

        local_file_path = os.path.join(CONTENT_DIR, filepath)

        local_file_hash = get_file_hash(local_file_path)
        index_file_name = f"{filepath}.{embedding_model}.{local_file_hash}.index.json"

        local_index_path = os.path.join(CONTENT_DIR, index_file_name)

        if os.path.exists(local_index_path):
            models.chatpdf.load_index(local_index_path)
            index_path = local_index_path
            file_status = "文件已成功加载，请开始提问"

        elif os.path.exists(local_file_path):
            models.chatpdf.load_pdf_file(local_file_path)
            models.chatpdf.save_index(local_index_path)
            index_path = local_index_path
            if index_path:
                file_status = "文件索引并成功加载，请开始提问"
            else:
                file_status = "文件未成功加载，请重新上传文件"
    else:
        file_status = "模型未完成加载，请先在加载模型后再导入文件"

    return index_path, history + [[None, file_status]]


def reset_chat(chatbot, state):
    return None, None


init_message = """欢迎使用 ChatPDF Web UI，可以直接提问或上传文件后提问 """


def chat_ui(embedding_model):
    index_path, file_status, model_status = gr.State(""), gr.State(""), gr.State("")
    file_list = gr.State(get_file_list())

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot([[None, init_message], [None, None]],
                                 elem_id="chat-box",
                                 show_label=False).style(height=700)
            query = gr.Textbox(
                show_label=False,
                placeholder="请输入提问内容，按回车进行提交",
            ).style(container=False)
            clear_btn = gr.Button('🔄Clear!', elem_id='clear').style(full_width=True)

        with gr.Column(scale=1):
            with gr.Row():
                chat_mode = gr.Radio(choices=["chat", "pdf"], value="pdf", label="聊天模式")

            with gr.Row():
                topn = gr.Slider(1, 100, 20, step=1, label="最大搜索数量")
                max_input_size = gr.Slider(512, 4096, MAX_INPUT_LEN, step=10, label="摘要最大长度")

            with gr.Tab("select"):
                with gr.Row():
                    selectFile = gr.Dropdown(
                        file_list.value,
                        label="content file",
                        interactive=True,
                        value=file_list.value[0] if len(file_list.value) > 0 else None
                    )
                    # get_file_list_btn = gr.Button('🔄').style(width=10)
            with gr.Tab("upload"):
                file = gr.File(
                    label="content file",
                    file_types=['.txt', '.md', '.docx', '.pdf']
                )
            load_file_button = gr.Button("加载文件")

    # 将上传的文件保存到content文件夹下,并更新下拉框
    file.upload(
        upload_file,
        inputs=[file, file_list],
        outputs=[selectFile, file_list]
    )
    load_file_button.click(
        get_vector_store,
        show_progress=True,
        inputs=[selectFile, chatbot, embedding_model],
        outputs=[index_path, chatbot],
    )
    query.submit(
        get_answer,
        [query, index_path, chatbot, topn, max_input_size, chat_mode],
        [chatbot, query],
    )
    clear_btn.click(reset_chat, [chatbot, query], [chatbot, query])
