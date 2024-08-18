#!/usr/bin/env python
# coding: utf-8

# In[29]:


from openai_client import call_openai_api


# In[8]:


estore_data = {
    "products": [
        {
            "id": 1,
            "name": "wireless mouse",
            "price": 25.99,
            "quantity_available": 100,
            "items_sold": 150
        },
        {
            "id": 2,
            "name": "mechanical keyboard",
            "price": 79.99,
            "quantity_available": 50,
            "items_sold": 75
        },
        {
            "id": 3,
            "name": "usb-c hub",
            "price": 34.99,
            "quantity_available": 2,
            "items_sold": 120
        }
    ]
}


# ### Converting the python functions as langchain tool

# #### simple tool decorator

# name	str	Must be unique within a set of tools provided to an LLM or agent.
# description	str	Describes what the tool does. Used as context by the LLM or agent.
# args_schema	Pydantic BaseModel	Optional but recommended, can be used to provide more information (e.g., few-shot examples) or validation for expected parameters

# In[9]:


from langchain_core.tools import tool

@tool
def buy_product(product_name):
    """   
    the stock availale is updated with each purchase
    returns a confirmation message to the user or a regret message when item is not in stock. 
    """
    for product in estore_data['products']:
        if product['name'] == product_name.lower():
            if product['quantity_available'] > 0:
                product['quantity_available'] -= 1
                product['items_sold'] += 1
                return f"{product_name} purchase confirmed. Remaining stock {product['quantity_available']}"
            else:
                return f"Regret, {product_name} is not in stock."
    return f"Product {product_name} not found."


# #### type of arguments has not been detected as str, lets be more sprcific

# In[10]:


@tool
def buy_product(product_name:str )->str:
    """   
    the stock availale is updated with each purchase
    returns a confirmation message to the user or a regret message when item is not in stock. 
    """
    for product in estore_data['products']:
        if product['name'] == product_name.lower():
            if product['quantity_available'] > 0:
                product['quantity_available'] -= 1
                product['items_sold'] += 1
                return f"{product_name} purchase confirmed. Remaining stock {product['quantity_available']}"
            else:
                return f"Regret, {product_name} is not in stock."
    return f"Product {product_name} not found."
# print("name:", buy_product.name)
# print("description:", buy_product.description)
# print("args schema:", buy_product.args_schema.schema())


# @tool can optionally parse Google Style docstrings and associate the docstring components (such as arg descriptions) to the relevant parts of the tool schema. To toggle this behavior, specify parse_docstring:
# 
# @tool(parse_docstring=True)
# def foo(bar: str, baz: int) -> str:
#     """The foo.
# 
#     Args:
#         bar: The bar.
#         baz: The baz.
#     """
#     return bar

# ####  Usig Annotation-  @tool supports parsing of annotations, nested schemas

# In[11]:


from typing import Annotated
@tool
def buy_product(
    product_name: Annotated[str, "name of the product"], 
    )->str:
    """   
    the stock availale is updated with each purchase
    returns a confirmation message to the user or a regret message when item is not in stock. 
    """
    for product in estore_data['products']:
        if product['name'] == product_name.lower():
            if product['quantity_available'] > 0:
                product['quantity_available'] -= 1
                product['items_sold'] += 1
                return f"{product_name} purchase confirmed. Remaining stock {product['quantity_available']}"
            else:
                return f"Regret, {product_name} is not in stock."
    return f"Product {product_name} not found."
# print("name:", buy_product.name)
# print("description:", buy_product.description)
# print("args schema:", buy_product.args_schema.schema())


# #### Using Pydantic 

# In[12]:


from langchain.pydantic_v1 import BaseModel, Field

class BuypoductInputSchema(BaseModel):
    product_name: str = Field(description="name of the product to be purchased")

@tool("buy_product-tool", args_schema=BuypoductInputSchema, return_direct=True) 
### return_direct returns the output directly instead of sending it back to model
def buy_product(product_name:str )->str:
    """   
    the stock availale is updated with each purchase
    returns a confirmation message to the user or a regret message when item is not in stock. 
    """
    for product in estore_data['products']:
        if product['name'] == product_name.lower():
            if product['quantity_available'] > 0:
                product['quantity_available'] -= 1
                product['items_sold'] += 1
                return f"{product_name} purchase confirmed. Remaining stock {product['quantity_available']}"
            else:
                return f"Regret, {product_name} is not in stock."
    return f"Product {product_name} not found."
# print("name:", buy_product.name)
# print("description:", buy_product.description)
# print("args schema:", buy_product.args_schema.schema())


# #### StructuredTool

# In[13]:


from langchain_core.tools import StructuredTool

def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


async def amultiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


calculator = StructuredTool.from_function(func=multiply, 
                                          # name="Calculator",
                                          # description="multiply numbers",
                                          # args_schema=CalculatorInput, ## pydantic class
                                          coroutine=amultiply)

# print(calculator.invoke({"a": 2, "b": 3}))
# print(await calculator.ainvoke({"a": 2, "b": 5}))


# #### Subclass BaseTool

# In[14]:


# https://api.python.langchain.com/en/latest/tools/langchain_core.tools.BaseTool.html
from typing import Optional, Type
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
buy_product_description = """ the stock availale is updated with each purchase.
    returns a confirmation message to the user or a regret message when item is not in stock. 
    """

class BuyProductTool(BaseTool):
    name = "buy_product"
    description = buy_product_description
    args_schema: Type[BaseModel] = BuypoductInputSchema  ## created previously using pydantic
    return_direct: bool = True

    def _run(self, product_name: str, 
             run_manager: Optional[CallbackManagerForToolRun]=None,
            ) -> str:
        for product in estore_data['products']:
            if product['name'] == product_name.lower():
                if product['quantity_available'] > 0:
                    product['quantity_available'] -= 1
                    product['items_sold'] += 1
                    return f"{product_name} purchase confirmed. Remaining stock {product['quantity_available']}"
                else:
                    return f"Regret, {product_name} is not in stock."
        return f"Product {product_name} not found."

    async def _arun(self, product_name: str, 
             run_manager: Optional[AsyncCallbackManagerForToolRun]=None,
            ) -> str:
         # If the calculation is cheap, you can just delegate to the sync implementation
         # as shown below.
         # If the sync calculation is expensive, you should delete the entire _arun method.
         # LangChain will automatically provide a better implementation that will
         # kick off the task in a thread to make sure it doesn't block other async code.
         return self._run(product_name, run_manager=run_manager.get_sync())

buyproduct_tool = BuyProductTool()
# print("name:", buyproduct_tool.name)
# print("description:", buyproduct_tool.description)
# print("args schema:", buyproduct_tool.args)   
# print('invoking sync ...:', buyproduct_tool.invoke('mechanical keyboard'))
# print('invoking async ...:', await  buyproduct_tool.ainvoke('mechanical keyboard'))


# #### Creating tools from Runnables

# In[15]:


from langchain_core.language_models import GenericFakeChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [("human", "Hello. Please respond in the style of {answer_style}.")]
)

# Placeholder LLM
llm = GenericFakeChatModel(messages=iter(["hello matey"]))

chain = prompt | llm | StrOutputParser()

as_tool = chain.as_tool(
    name="Style responder", description="Description of when to use tool."
)
as_tool.args


# #### tool for the calculate revenue with handle_tool_error exception

# Sometimes there are artifacts of a tool's execution that we want to make accessible to downstream components in our chain or agent, but that we don't want to expose to the model itself. For example if a tool returns custom objects like Documents, we may want to pass some view or metadata about this output to the model without passing the raw output to the model. At the same time, we may want to be able to access this full output elsewhere, for example in downstream tools.
# The Tool and ToolMessage interfaces make it possible to distinguish between the parts of the tool output meant for the model (this is the ToolMessage.content) and those parts which are meant for use outside the model (ToolMessage.artifact).
# If we invoke our tool with a ToolCall (like the ones generated by tool-calling models), we'll get back a ToolMessage that contains both the content and artifact generated by the Tool.
# REQUIRES langchain-core >= 0.2.19

# In[16]:


from langchain_core.tools import ToolException
from typing import Union, Callable, Literal

class CalcRevInputSchema(BaseModel):
    product_name: str = Field(description="name of the product to be purchased")

calculate_revenue_description = """ returns revenue for the product or total revenue when no product name is mentione
    """

class CalculateRevenueTool(BaseTool):
    name = "calculate_revenue"
    description = calculate_revenue_description
    args_schema: Type[BaseModel] = CalcRevInputSchema
    return_direct: bool = True ## the AgentExecutor will stop looping, output does not go to model
    # handle_tool_error = True ## or a string message or a callable function. will not work. use typing 
    handle_tool_error: Optional[Union[bool, str, Callable[[ToolException], str]]] = True
    response_format: Literal['content', 'content_and_artifact'] = 'content_and_artifact' ## or content
    ## metadata: Optional[Dict[str, Any]] = None , ###Optional metadata associated with the tool. Defaults to None. This metadata will be associated with each call to this tool, and passed as arguments to the handlers defined in callbacks. You can use these to eg identify a specific instance of a tool with its use case.
    verbose: bool = True

    
    def _run(self, product_name: str=None, 
             run_manager: Optional[CallbackManagerForToolRun]=None,
            ) -> str:
        total_revenue = 0.0
        artifact_prodct_wise_rev = {}
        for product in estore_data['products']:
            artifact_prodct_wise_rev[product["name"]] = product['items_sold'] * product['price']
        for product in estore_data['products']:
            if product_name and product['name'] == product_name.lower():
                content = f"Revenue for {product_name}: ${product['items_sold'] * product['price']:.2f}"
                return content, artifact_prodct_wise_rev
            total_revenue += product['items_sold'] * product['price']
        if product_name is None:
            content = f"Total revenue: ${total_revenue:.2f}"
            return content, artifact_prodct_wise_rev
        # return f"Product {product_name} not found." instead we will use ToolException        
        raise ToolException(f"Product {product_name} not found.")

    async def _arun(self, product_name: str=None, 
             run_manager: Optional[AsyncCallbackManagerForToolRun]=None,
            ) -> str:         
         return self._run(product_name, run_manager=run_manager.get_sync())

calculate_revenue_tool = CalculateRevenueTool()
# print("name:", calculate_revenue_tool.name)
# print("description:", calculate_revenue_tool.description)
# print("args schema:", calculate_revenue_tool.args)
# print('invoking sync with ToolCall format to get ToolMessage ...:' )
# tool_message = calculate_revenue_tool.invoke(
#     {
#         "name": "calculate_revenue_tool",
#         "args": {"product_name": "mechanical keyboard"},
#         "id": "1" , ## required field
#         "type": "tool_call", ## required field
#     }
# )     
# print('tool_message ...:', tool_message)
# print('invoking async ...:', await  calculate_revenue_tool.ainvoke('mechanical keyboard'))
# print('handle_tool_error ToolException sync ...:', calculate_revenue_tool.invoke('mechanical keyboard'))


# ### Use the tool inside prompt, instead of hard coded prompt

# In[17]:


calculate_revenue_tool.args


# In[18]:


args_list = list(calculate_revenue_tool.args.keys())
args_list


# In[ ]:





# In[19]:


from typing import List
# tools = [CalculateRevenueTool(), BuyProductTool()]
tools = [buyproduct_tool, calculate_revenue_tool]

def generate_action_prompt(tools: List):
    action_list_txt = ""
    for tool in tools:
        args_list = list(tool.args.keys())
        args_as_text = ", ".join(args_list)
        action_signature = f"{tool.name}: {args_as_text} \n"
        action_description = f"{tool.description}\n\n"
        action_list_txt += action_signature + action_description
    return action_list_txt
action_list_txt = generate_action_prompt(tools)
# print(action_list_txt)        


# In[20]:


react_instruction_str = """You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Your available actions are:"""

example_session_str = """Example session:

Question: I would like to buy a wireless mouse, can you despatch it please ?
Thought: I should buy the wireless mouse using the buy_product action
Action: buy_product: wireless mouse
PAUSE

You will be called again with this:

Observation: wireless mouse purchase confirmed. Remaining stock 40.

You then output:

Answer: Thanks for your order. Your wireless mouse has been despatched to you. 
when the stock is not there you answer should be changed accordingly"""


# In[21]:


sys_template_string_agent = """
{react_instruction_str}

{action_list_txt}

{example_session_str}
"""
agent_react_prompt_with_var = sys_template_string_agent.format(
    react_instruction_str = react_instruction_str,
    action_list_txt=action_list_txt,
    example_session_str=example_session_str).strip()
# print(agent_react_prompt_with_var)


# In[23]:


def autoagent(question, prompt=None,  max_turns=5):
    i = 0
    if prompt:
        agent = Agent(prompt)
    agent = Agent(agent_react_prompt)
    user_prompt = question
    while i < max_turns:
        i += 1
        result = agent(user_prompt)
        print(result)
        parse_result = parse_action_name_n_params(result)
        if isinstance( parse_result, tuple):
            action, action_input = parse_result
            if action not in known_actions:
                raise Exception("Unknown actions: {}: {}".format(action, action_input))
            print(f'calling {action}: with arguments: {action_input}')            
            observation = known_actions[action](action_input)
            print('observation:', observation)
            user_prompt = "Observation: {}".format(observation)
        else:
            return    


# In[25]:


class Agent:
    def __init__(self, system=""):
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        completion = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            temperature=0,
            messages=self.messages)
        return completion.choices[0].message.content


# In[30]:


agent_react_prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Your available actions are:

buy_product:
e.g. buy_product: product_name
the stock availale is updated with each purchase
returns a confirmation message to the user or a regret message when item is not in stock. 

calculate_revenue:
e.g. calculate_revenue: product_name
returns revenue for the product or total revenue when no product name is mentioned

Example session:

Question: I would like to buy a wireless mouse, can you despatch it please ?
Thought: I should buy the wireless mouse using the buy_product action
Action: buy_product: wireless mouse
PAUSE

You will be called again with this:

Observation: wireless mouse purchase confirmed. Remaining stock 40.

You then output:

Answer: Thanks for your order. Your wireless mouse has been despatched to you. 
when the stock is not there you answer should be changed accordingly
""".strip()


# In[38]:


import os
import json
import re
import load_configs
import importlib
importlib.reload(load_configs)
from load_configs import (
    openai_api_key,
    llama_api_key, ai21_api_key, 
    gemini_api_key
)
from openai_client import call_openai_api
# from langchain_community.llms import OpenAI
from  openai import OpenAI
client =OpenAI(api_key=openai_api_key)


# In[45]:


def parse_action_name_n_params(result):
    result.split('\n')
    pattern_action = re.compile('^Action: (\w+): (.*)$')
    parsed_actions = []
    for line in result.split('\n'):
        action_found = pattern_action.match(line)
        if action_found:
            parsed_actions.append(action_found)
    print(parsed_actions)
    if parsed_actions:
        action, action_input = parsed_actions[0].groups()
        print(action, action_input )
        return action, action_input
    return
# action, action_input = parse_action_name_n_params(result)


# In[ ]:





# In[43]:


# E-store data
# estore_data = {
#     "products": [
#         {
#             "id": 1,
#             "name": "wireless mouse",
#             "price": 25.99,
#             "quantity_available": 100,
#             "items_sold": 150
#         },
#         {
#             "id": 2,
#             "name": "mechanical keyboard",
#             "price": 79.99,
#             "quantity_available": 50,
#             "items_sold": 75
#         },
#         {
#             "id": 3,
#             "name": "usb-c hub",
#             "price": 34.99,
#             "quantity_available": 2,
#             "items_sold": 120
#         }
#     ]
# }

def buy_product(product_name):
    for product in estore_data['products']:
        if product['name'] == product_name.lower():
            if product['quantity_available'] > 0:
                product['quantity_available'] -= 1
                product['items_sold'] += 1
                return f"{product_name} purchase confirmed. Remaining stock {product['quantity_available']}"
            else:
                return f"Regret, {product_name} is not in stock."
    return f"Product {product_name} not found."

def calculate_revenue(product_name=None):
    total_revenue = 0.0
    for product in estore_data['products']:
        if product_name and product['name'] == product_name.lower():
            return f"Revenue for {product_name}: ${product['items_sold'] * product['price']:.2f}"
        total_revenue += product['items_sold'] * product['price']
    if product_name is None:
        return f"Total revenue: ${total_revenue:.2f}"
    return f"Product {product_name} not found."

# Example usage:
# Buying a product
# print(buy_product("Wireless Mouse"))  # Wireless Mouse has been dispatched.
# print(buy_product("Mechanical Keyboard"))  # Mechanical Keyboard has been dispatched.
# print(buy_product("USB-C Hub"))  # USB-C Hub has been dispatched.
# print(buy_product("Smartphone"))  # Product Smartphone not found.

# # Calculating revenue
# print(calculate_revenue("Wireless Mouse"))  # Revenue for Wireless Mouse: $4158.50
# print(calculate_revenue("Mechanical Keyboard"))  # Revenue for Mechanical Keyboard: $5992.50
# print(calculate_revenue())  # Total revenue: $17086.50
# print(calculate_revenue("Smartphone"))  # Product Smartphone not found.
known_actions = {
    "buy_product": buy_product,
    "calculate_revenue": calculate_revenue
}


# In[44]:


# question = """can you buy me one wireless mouse and one usb-c hub ? """
# autoagent(question, prompt=agent_react_prompt_with_var)


# In[ ]:


# llm.bind_tools


# For the models that use tool calling, no special prompting is needed. ???

# ###  Understanding Langchain Prompt

# In[ ]:


# print(ChatPromptTemplate.__doc__)


# In[ ]:


# get_ipython().system('pip list|grep langchain')


# In[ ]:


from langchain.prompts import ChatPromptTemplate


# In[ ]:


template = ChatPromptTemplate([
                ("system", "You are a helpful AI bot."),
                # Means the template will receive an optional list of messages under
                # the "conversation" key
                ("placeholder", "{conversation}")
                # Equivalently:
                # MessagesPlaceholder(variable_name="conversation", optional=True)
            ])
# print(template)
# prompt_value = template.invoke(
#     {
#         "conversation": [
#             ("human", "Hi!"),
#             ("ai", "How can I assist you today?"),
#             ("human", "Can you make me an ice cream sundae?"),
#             ("ai", "No.")
#         ]
#     }
# )
# prompt_value


# In[ ]:


template = ChatPromptTemplate([
    ("system", "You are a helpful AI bot. Your name is {bot_name}."),
    ("human", "Hello, how are you doing?"),
    ("ai", "{greetings}, I'm doing well, thanks!"),
    ("human", "{user_input}"),
    ],
    input_variables=['user_input'],
    optional_variables=["greetings"],
    partial_variables={"bot_name": "Monalisa"}                            

)
# print(template)
final_input = {            
            "user_input": "What is your name?"
        }
# try:
#     prompt_value = template.invoke(final_input)
# except Exception as e:
#     print(e)
# # print(prompt_value)
# template_partial = template.partial(bot_name="Monalisa")
# print(template_partial)
# try:
#     template_partial.invoke(final_input)
# except Exception as e:
#     print(e)


# In[ ]:


from langchain_core.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("user", "Tell me a joke about {topic}")
])

# prompt_template.invoke({"topic": "cats"})


# In[ ]:


from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

prompt_template = ChatPromptTemplate.from_messages([
    # ("system", "You are a helpful assistant"),
    SystemMessage(content='You are a helpful assistant'),
    MessagesPlaceholder("msgs")
])

# prompt_template.invoke({"msgs": [HumanMessage(content="hi!"), HumanMessage(content="How are you")]})


# In[ ]:


from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate


# In[ ]:


lc_sys_prompt_agent = SystemMessagePromptTemplate.from_template(sys_template_string_agent)
lc_sys_prompt_agent


# In[ ]:


# print(lc_sys_prompt_agent.prompt.input_variables)
# print(lc_sys_prompt_agent.prompt.template)
# print(lc_sys_prompt_agent.format(
#     react_instruction_str = react_instruction_str,
#     action_list_txt=action_list_txt,
#     example_session_str=example_session_str))


# In[ ]:





# In[ ]:


class Agent:
    def __init__(self, system=""):
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        completion = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            temperature=0,
            messages=self.messages)
        return completion.choices[0].message.content


# https://lilianweng.github.io/posts/2023-06-23-agent/

# In[ ]:





# In[ ]:




