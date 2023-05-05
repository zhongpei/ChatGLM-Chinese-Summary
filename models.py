import os

from loguru import logger

from utils.chatpdf import ChatPDF
from utils.llm import LLM
from utils.singleton import Singleton

MAX_INPUT_LEN = 2048

embedding_model_dict = {
    "text2vec-large": "GanymedeNil/text2vec-large-chinese",
    "text2vec-base": "shibing624/text2vec-base-chinese",
    "sentence-transformers": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
}

# supported LLM models
llm_model_dict = {
    #"chatglm-6b": "E:\\sdwebui\\image2text_prompt_generator\\models\\chatglm-6b",
    #"chatglm-6b-int4": "E:\\sdwebui\\image2text_prompt_generator\\models\\chatglm-6b-int4",
    "chatglm-6b": "THUDM/chatglm-6b",
    "chatglm-6b-int4": "THUDM/chatglm-6b-int4",
    "llama-7b": "decapoda-research/llama-7b-hf",
    "llama-13b": "decapoda-research/llama-13b-hf",
    "t5-lamini-flan-783M": "MBZUAI/LaMini-Flan-T5-783M",
}

llm_model_dict_list = list(llm_model_dict.keys())
embedding_model_dict_list = list(embedding_model_dict.keys())


@Singleton
class Models(object):

    def __init__(self):
        self._chatpdf = None
        self._llm_model = None

    def is_active(self):
        return self._chatpdf is not None and self._llm_model is not None

    @property
    def chatpdf(self):
        return self._chatpdf

    @property
    def llm_model(self):
        return self._llm_model

    def reset_model(self):
        if self._chatpdf is not None:
            del self._chatpdf

        if self._llm_model is not None:
            del self._llm_model

        self._chatpdf = None
        self._llm_model = None

    def init_model(self, llm_model, llm_lora, embedding_model):
        try:
            self.reset_model()

            llm_lora_path = None
            if llm_lora is not None and os.path.exists(llm_lora):
                llm_lora_path = llm_lora
            self._chatpdf = ChatPDF(
                sim_model_name_or_path=embedding_model_dict.get(
                    embedding_model,
                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                ),

            )
            self._llm_model = LLM(
                gen_model_type=llm_model.split('-')[0],
                gen_model_name_or_path=llm_model_dict.get(llm_model, "THUDM/chatglm-6b-int4"),
                lora_model_name_or_path=llm_lora_path
            )
            if self._chatpdf is not None and self._llm_model is not None:
                model_status = f"模型{llm_model} lora:{llm_lora} embedding:{embedding_model}已成功加载"
            else:
                model_status = f"llm:{self._llm_model} pdf:{self._chatpdf}加载失败"
            logger.info(model_status)
            return model_status
        except Exception as e:
            logger.error(f"加载模型失败:{e}")
            raise e


models = Models.instance()
