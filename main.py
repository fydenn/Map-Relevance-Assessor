from agent import AI_agent

def get_request():
    return input("\nУкажите запрос и название организации: ")

def print_label(label):
    print(f"\n{label}")


def main():
    agent = AI_agent()
    
    req = get_request()
    label = agent.get_label(text= req)
    print_label(label)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('\nпока!')

