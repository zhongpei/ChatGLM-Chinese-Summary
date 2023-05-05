from similarities import Similarity
from textgen import ChatGlmModel, LlamaModel
from transformers import pipeline
from loguru import logger
import torch

device_id = 0 if torch.cuda.is_available() else -1

PROMPT_TEMPLATE = """\
基于以下已知信息，简洁和专业的来回答用户的问题。
如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请使用中文。

已知内容:
{context_str}

问题:
{query_str}
"""


class ChatPDF:
    def __init__(
            self,
            sim_model_name_or_path: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",

    ):
        self.sim_model = Similarity(model_name_or_path=sim_model_name_or_path)

        self.history = None
        self.pdf_path = None

    def load_pdf_file(self, pdf_path: str):
        """Load a PDF file."""
        if pdf_path.endswith('.pdf'):
            corpus = self.extract_text_from_pdf(pdf_path)
        elif pdf_path.endswith('.docx'):
            corpus = self.extract_text_from_docx(pdf_path)
        elif pdf_path.endswith('.md'):
            corpus = self.extract_text_from_markdown(pdf_path)
        else:
            corpus = self.extract_text_from_txt(pdf_path)
        self.sim_model.add_corpus(corpus)
        self.pdf_path = pdf_path

    @staticmethod
    def extract_text_from_pdf(file_path: str):
        """Extract text content from a PDF file."""
        import PyPDF2
        contents = []
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                page_text = page.extract_text().strip()
                raw_text = [text.strip() for text in page_text.splitlines() if text.strip()]
                new_text = ''
                for text in raw_text:
                    new_text += text
                    if text[-1] in ['.', '!', '?', '。', '！', '？', '…', ';', '；', ':', '：', '”', '’', '）', '】', '》', '」',
                                    '』', '〕', '〉', '》', '〗', '〞', '〟', '»', '"', "'", ')', ']', '}']:
                        contents.append(new_text)
                        new_text = ''
                if new_text:
                    contents.append(new_text)
        return contents

    @staticmethod
    def extract_text_from_txt(file_path: str):
        """Extract text content from a TXT file."""
        contents = []
        with open(file_path, 'r', encoding='utf-8') as f:
            contents = [text.strip() for text in f.readlines() if text.strip()]
        return contents

    @staticmethod
    def extract_text_from_docx(file_path: str):
        """Extract text content from a DOCX file."""
        import docx
        document = docx.Document(file_path)
        contents = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
        return contents

    @staticmethod
    def extract_text_from_markdown(file_path: str):
        """Extract text content from a Markdown file."""
        import markdown
        from bs4 import BeautifulSoup
        with open(file_path, 'r', encoding='utf-8') as f:
            markdown_text = f.read()
        html = markdown.markdown(markdown_text)
        soup = BeautifulSoup(html, 'html.parser')
        contents = [text.strip() for text in soup.get_text().splitlines() if text.strip()]
        return contents

    @staticmethod
    def _add_source_numbers(lst):
        """Add source numbers to a list of strings."""
        return [f'[{idx + 1}]\t "{item}"' for idx, item in enumerate(lst)]

    def query(
            self,
            llm_model,
            query,
            topn: int = 5,
            max_length: int = 1024,
            max_input_size: int = 1024,
            use_history: bool = False,

    ):
        """Query from corpus."""

        sim_contents = self.sim_model.most_similar(query, topn=topn)

        reference_results = []
        for query_id, id_score_dict in sim_contents.items():
            for corpus_id, s in id_score_dict.items():
                reference_results.append(self.sim_model.corpus[corpus_id])
        if not reference_results:
            return '没有提供足够的相关信息', reference_results
        reference_results = self._add_source_numbers(reference_results)

        context_str = '\n'.join(reference_results)[:(max_input_size - len(PROMPT_TEMPLATE))]

        if use_history:
            response, out_history = llm_model.generate_answer(query, context_str, self.history, max_length=max_length)
            self.history = out_history
        else:
            response, out_history = llm_model.generate_answer(query, context_str)

        return response, out_history, reference_results

    def save_index(self, index_path=None):
        """Save model."""
        if index_path is None:
            index_path = '.'.join(self.pdf_path.split('.')[:-1]) + '_index.json'
        self.sim_model.save_index(index_path)

    def load_index(self, index_path=None):
        """Load model."""
        if index_path is None:
            index_path = '.'.join(self.pdf_path.split('.')[:-1]) + '_index.json'
        self.sim_model.load_index(index_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2:
        gen_model_name_or_path = sys.argv[1]
    else:
        print('Usage: python chatpdf.py <gen_model_name_or_path>')
        gen_model_name_or_path = "THUDM/chatglm-6b-int4"
    m = ChatPDF(gen_model_name_or_path=gen_model_name_or_path)
    m.load_pdf_file(pdf_path='sample.pdf')
    response = m.query('自然语言中的非平行迁移是指什么？')
    print(response[0])
    response = m.query('本文作者是谁？')
    print(response[0])
