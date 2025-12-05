from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-1.5-flash")

chat_history=[]
sys_msg=SystemMessage(content="You are a english teacher")
print("System Message:",sys_msg)
chat_history.append(sys_msg)
while True:
    user_input=input("You : ")
    if(user_input=="exit"):break
    human_msg=HumanMessage(content=user_input)
    chat_history.append(human_msg)
    res=model.invoke(chat_history)
    print(res.content)
    ai_msg=AIMessage(content=res.content)
    chat_history.append(ai_msg)


print(chat_history)
