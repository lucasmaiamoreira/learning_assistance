import streamlit as st
import requests
import json
import time
from langchain.chat_models.base import BaseChatModel
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import SequentialChain, LLMChain
from langchain.schema import BaseMessage, AIMessage, HumanMessage, LLMResult, ChatGeneration, ChatResult
import logging
from typing import List, Optional


class OllamaChat(BaseChatModel):
    url: str
    headers: dict
    model_name: str

    def get_prompt(self, messages: List[BaseMessage]) -> str:
        prompt = f"{messages[0].content} "
        for i, message in enumerate(messages[1:]):
            if isinstance(message, HumanMessage):
                prompt += f"USUÁRIO: {message.content} "
            elif isinstance(message, AIMessage):
                prompt += f"ASSISTENTE: {message.content}</s>"
        prompt += f"ASSISTENTE:"
        return prompt

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None) -> LLMResult:
        prompt = self.get_prompt(messages)
        payload = {
            "model": self.model_name,
            "prompt": prompt
        }
        responses = []
        try:
            response = requests.post(self.url, headers=self.headers, data=json.dumps(payload), stream=True)
            
            generated_text = ""
            placeholder = st.empty()

            for chunk in response.iter_lines():
                if chunk:
                    decoded_chunk = chunk.decode('utf-8')
                    try:
                        parsed_obj = json.loads(decoded_chunk)
                        if 'response' in parsed_obj:
                            response_part = parsed_obj['response']
                            generated_text += response_part
                            placeholder.markdown(generated_text.strip())
                            time.sleep(0.05)
                    except json.JSONDecodeError as e:
                        print(f"Erro ao decodificar JSON: {e}")

            generated_text = generated_text.strip()
            
        except requests.RequestException as e:
            generated_text = f"Erro na solicitação: {str(e)}"
        
        ai_message = AIMessage(content=generated_text)
        chat_result = ChatResult(generations=[ChatGeneration(message=ai_message)])
        
        return chat_result

    @property
    def _llm_type(self) -> str:
        return self.model_name

    def _agenerate(self):
        return None

# Inicializando o modelo
url = "http://localhost:11434/api/generate"
headers = {"Content-Type": "application/json"}
model_name = "llama3"

chat = OllamaChat(url=url, headers=headers, model_name=model_name)

class GeneralEducationTemplate:
    def __init__(self):
        self.system_template = """
        Você é um professor altamente qualificado com experiência em uma ampla gama de tópicos acadêmicos e educacionais. Sua tarefa é responder a perguntas gerais dos usuários, fornecendo explicações claras e detalhadas. Além disso, você deve criar quizzes interativos para ajudar os usuários a testar e aprofundar seu conhecimento sobre qualquer assunto que eles desejarem aprender.
        Você deve lembrar as perguntas feitas criando uma memória dentro do chat, não precisa dizer isto ao usuário. Deve responder sempre no mesmo idioma que for realizado a pergunta.

        **Diretrizes para suas respostas:**
        - **Clareza**: Sempre procure explicar os conceitos de forma clara e compreensível.
        - **Detalhamento**: Forneça detalhes suficientes para cobrir o tópico solicitado sem sobrecarregar o usuário com informações excessivas.
        - **Exemplos**: Sempre que possível, inclua exemplos práticos para ilustrar os conceitos.
        - **Interatividade**: Quando criar quizzes, ofereça perguntas que desafiem o conhecimento do usuário e forneça feedback sobre as respostas corretas e incorretas.
        - **Tom**: Mantenha um tom amigável e encorajador, como se você estivesse conversando diretamente com o usuário.

        Por exemplo, se um usuário pergunta sobre "como funciona a fotossíntese", sua resposta deve explicar o processo em termos simples, talvez começando com uma breve definição, seguida pelos principais passos da fotossíntese, e terminando com um exemplo de como as plantas utilizam este processo para crescer. Se solicitado, você também pode criar um pequeno quiz para testar a compreensão do usuário sobre o tema.
        """
        
        self.human_template = """
        ####{request}####
        """
        self.system_message_prompt = SystemMessagePromptTemplate.from_template(self.system_template)
        self.human_message_prompt = HumanMessagePromptTemplate.from_template(self.human_template, input_variables=["request"])
        self.chat_prompt = ChatPromptTemplate.from_messages([self.system_message_prompt, self.human_message_prompt])

class Agent:
    def __init__(self, chat_model, verbose=True):
        self.logger = logging.getLogger(__name__)
        if verbose:
            self.logger.setLevel(logging.INFO)
        self.chat_model = chat_model
        self.verbose = verbose

    def get_response(self, request):
        template = GeneralEducationTemplate()
        education_chain = LLMChain(
            llm=self.chat_model,
            prompt=template.chat_prompt,
            verbose=self.verbose,
            output_key='response'
        )
        overall_chain = SequentialChain(
            chains=[education_chain],
            input_variables=["request"],
            output_variables=["response"],
            verbose=self.verbose
        )

        try:
            result = overall_chain({"request": request}, return_only_outputs=True)
        
            if 'response' not in result or not isinstance(result['response'], str):
                raise ValueError(f"Expected result to have 'response' key with a string value, but got {result}")
        
            return result
        except Exception as e:
            self.logger.error(f"Error in get_response: {e}")
            return {"response": "Desculpe, ocorreu um erro ao processar sua solicitação."}

my_agent = Agent(chat)

st.set_page_config(page_title="App de Auxílio em Aprendizagem")
st.title("App de Auxílio em Aprendizagem 📚")

st.sidebar.header("Parâmetros ⚙️")
temperature = st.sidebar.slider("Temperatura: Controla a aleatoriedade das respostas geradas pelo modelo!", min_value=0.01, max_value=1.00, value=0.10, step=0.01)
max_length = st.sidebar.slider("Comprimento máximo: Define o comprimento máximo da resposta gerada.", min_value=32, max_value=128, value=120, step=1)

if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state['chat_history'] = [{"role": "assistant", "content": "Como posso ajudar você hoje? Pergunte-me qualquer coisa!"}]

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = [{"role": "assistant", "content": "Como posso ajudar você hoje? Pergunte-me qualquer coisa!"}]

for chat in st.session_state['chat_history']:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

if prompt := st.chat_input("Digite sua pergunta ou tópico de interesse:"):
    st.session_state['chat_history'].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = my_agent.get_response(prompt)["response"]
    
        st.session_state['chat_history'].append({"role": "assistant", "content": response})

