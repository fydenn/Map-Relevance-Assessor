import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import  SystemMessage
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode

class AI_agent():
    def __init__(self):
        # Load environment variables
        load_dotenv('.env')
        OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        OPENAI_URL = os.getenv('OPENAI_URL')
        TAVILI_API_KEY = os.getenv("TAVILI_API_KEY")

        # Initialize tools
        self.search_tool = TavilySearch(
            max_results=10, 
            include_answer=True, 
            include_raw_content=False,
            tavily_api_key=TAVILI_API_KEY
        )

        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=OPENAI_API_KEY,
            openai_api_base=OPENAI_URL,
            temperature=0
        )
        
        # Compile the workflow
        self.app = self.compile_workflow()

    def should_continue(self, state: MessagesState):
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return "classification"    
    
    def get_info_node(self, state: MessagesState):
     
        messages = state["messages"]
        messages.append(SystemMessage(content='''
            Ты – помощник по оценке релевантности организаций. 
            Твоя задача - обработать запрос вида('общий запрос, название или названия одной организации') 
            собрать данные об организации и найти ключевые слова в общем запросе. 

            Тебе нужно собрать об организации и запросе следующие данные:
            запрос:
                * насколько общий или узкий это запрос(есть ли какие то уточнения в запросе, если есть, то запомнить, если нет, то не надо).

            организация:
                * Общее описание организации.
                * адрес.
                * товары и услуги, которые предоставляет.
                * если есть, то специализацию этой организации.

            Для поиска данных об организации для каждого пункта делай отдельный запрос с помощью инструмента TavilySearchResults.
            старайся находить связть общий запрос - организация.

                Ответь JSON в формате 
            {{"case_data": ключевые слова в общем запросе, все данные об организации в текстовом виде}}.
            Никакий других данных в ответе быть не должно, 
            только JSON, который можно распарсить

            '''))

        response = self.llm.invoke(messages)
    
        return {"messages": [response]}
    

    def classification_node(self, state: MessagesState):
        messages = state['messages']
        messages.append(SystemMessage(content='''
            Ты – помощник по оценке релевантности организаций. 
            Ты на вход получил json файл, где запрос вида('общий запрос'.'названия одной организации'), все остальное это описание организации.
            
            Твоя задача - на основе полученного описания, дать оценку релевантности организации для общего запроса пользователя,
            организация является релевантной, если она может предоставить услугу или товар, которые можно будет преобрести или воспользоваться по общему запросу,
            иначе эта организация не является релевантной.
                                      
            примеры:
                (сигары, кальянная) не релевантна, потому что в этой организации не продаются сигары.
                (Аэс, губниская ГЭС), не релевантна, потому что запрос был по поиску аэс, а не гэс.
                (Еда, сеть фастфуд) релевантна, потому что пользователь сможет найти там еду.
                (фубольный мяч, детский мир) релевантно, потому что в этом магазине можно купить футбольный мяч.
                                      
            ответь 1 - организация релевантна, а 0 - не релевантна.

            '''))
        
        response = self.llm.invoke(messages)

        return {"messages": [response]}

    def compile_workflow(self):
        tool_node = ToolNode([self.search_tool])

        workflow = StateGraph(MessagesState)

        workflow.add_node('info', self.get_info_node)
        workflow.add_node('tools', tool_node)
        workflow.add_node('classification', self.classification_node)

        workflow.set_entry_point('info')
        workflow.set_finish_point("classification")
        workflow.add_conditional_edges("info", 
                                    self.should_continue,     
                                    ["tools", 'classification'])
        workflow.add_edge('tools', 'info')
        return workflow.compile()

    def get_label(self, text: str):
        state_input = {'messages': text}
        result_state = self.app.invoke(state_input)
        return result_state['messages'][4].content
    
    
    