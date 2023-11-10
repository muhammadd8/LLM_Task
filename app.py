from utils.general import ConversationManager

conversation_manager = ConversationManager()
while True:
    user_input = input("User: ")
    if user_input.lower() == 'exit':
        print("Bot: Goodbye!")
        break
    response = conversation_manager.get_response(user_input)
    print("Bot:", response)
