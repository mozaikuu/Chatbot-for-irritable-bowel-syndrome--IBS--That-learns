import json
from difflib import get_close_matches

def load_knowledge_base(file_path: str)-> dict:
    with open(file_path, 'r') as file:
        data: dict = json.load(file)
    return data
  
def save_knowledge_base(file_path: str, data: dict)-> None:
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=2)

def find_best_match(user_question: str, questions: list[str])-> str | None:
    matches: list = get_close_matches(user_question, questions, n=1, cutoff=0.6)
    return matches[0] if matches else None
  
def get_answer_for_question(question: str, knowledge_base: dict)-> str | None:
    for q in knowledge_base["questions"]:
        if q["question"].lower() == question.lower():
            return q["answer"]
          
def chat_bot():
  knowledge_base: dict = load_knowledge_base("KnowledgeBase.json")
  
  print("Bot: Hi! I am a chatbot. I can answer questions about the following topics: 'irritable bowel syndrome' You can type 'exit' to leave the chat.")
  
  while True:
    user_input: str = input("You: ")
    if user_input.lower() == "exit":
        break
    best_match: str | None = find_best_match(user_input, [q["question"] for q in knowledge_base["questions"]])
    if best_match:
        answer: str | None = get_answer_for_question(best_match, knowledge_base)
        print(f"Bot: {answer}")
    else:
        print("Bot: i don't know the answer to that question. Can you please provide me with the answer?")
        new_answer: str = input("Type the answer or 'skip' to skip: ")
        
        if new_answer.lower() != "skip":
            knowledge_base["questions"].append({"question": user_input, "answer": new_answer})
            save_knowledge_base("KnowledgeBase.json", knowledge_base)
            print("Bot: Thanks for the new information. I will remember that for next time. would you like to ask me anything else?")
            
if __name__ == "__main__":
    chat_bot()