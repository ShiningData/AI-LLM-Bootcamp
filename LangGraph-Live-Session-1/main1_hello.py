from langgraph.graph import StateGraph, MessagesState, START, END
import random
from typing_extensions import TypedDict

class MyCustomState(MessagesState):
    rand_a: int = None
    bootcamp: str = None


email_precessor = StateGraph(MyCustomState)


def a1(state: MyCustomState):
    print("a1 in yapacağı işler")
    a = random.randint(1,10)
    bootcamp = 'VBO AILLM BC-3'
    return {'rand_a': a, 'bootcamp': bootcamp}

def a2(state: MyCustomState):
    print("a2 in yapacağı işler")
    print(state)

def a3(state: MyCustomState):
    print("a3 in yapacağı işler")

def a4(state: MyCustomState):
    print("a4 in yapacağı işler")
    print(state)

def after_a1_condition(state: MyCustomState):
    a = state['rand_a']
    if a%2==0:
        return 'a2'
    else:
        return 'a4'

email_precessor.add_node('a1', a1)
email_precessor.add_node('a2', a2)
email_precessor.add_node('a3', a3)
email_precessor.add_node('a4', a4)

email_precessor.add_edge(START, 'a1')
email_precessor.add_conditional_edges('a1', after_a1_condition)
email_precessor.add_edge('a2','a3')
email_precessor.add_edge('a4', 'a3')
email_precessor.add_edge('a4', END)

email_m_agent = email_precessor.compile()

result = email_m_agent.invoke(input={'messages': ['Hello']})

print(result)