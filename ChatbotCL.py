import argparse
import os
from ChatbotFunctions import (
    process_documents, 
    handle_userinput, 
    save_state, 
    load_state,
    temp_state_file
)

def main():
    """
    Main function to handle command-line arguments and execute appropriate actions.
    """
    parser = argparse.ArgumentParser(description="Chat with multiple PDFs and TXTs")
    parser.add_argument("--docs", nargs="+", help="List of documents to process")
    parser.add_argument("--add", nargs="+", help="List of documents to add to the existing session")
    parser.add_argument("--question", help="Question to ask about the documents")
    parser.add_argument("--save", help="Filename to save the state")
    parser.add_argument("--load", help="Filename to load the state from")

    args = parser.parse_args()

    conversation = None
    chat_history = []
    raw_text = ""
    filenames = []
    processed = False

    # Load state from a specified file
    if args.load:
        print("Loading state from file...")
        state = load_state(args.load, question_mode=bool(args.question), loading_session=True)
        if state:
            conversation = state['conversation']
            chat_history = state['chat_history']
            raw_text = state['raw_text']
            filenames = state.get('filenames', [])
        print("State loaded.")

    # Load temporary state if exists and no specific load file is provided
    if os.path.exists(temp_state_file) and not args.load:
        print("Loading temporary state...")
        state = load_state(temp_state_file, question_mode=bool(args.question))
        if state:
            conversation = state['conversation']
            chat_history = state['chat_history']
            raw_text = state['raw_text']
            filenames = state.get('filenames', [])
        print("Temporary state loaded.")

    # Process new documents
    if args.docs:
        print("Processing new documents...")
        conversation, raw_text, filenames = process_documents(args.docs)
        save_state(conversation, chat_history, raw_text, filenames, temp_state_file)
        processed = True
        print("Documents processed and state saved.")

    # Add documents to the existing session
    if args.add:
        print("Adding documents to the existing session...")
        conversation, new_raw_text, new_filenames = process_documents(args.add, raw_text)
        raw_text = new_raw_text
        filenames.extend(new_filenames)
        save_state(conversation, chat_history, raw_text, filenames, temp_state_file)
        processed = True
        print("Documents added and state saved.")

    # Handle user's question
    if args.question:
        if conversation:
            print("Handling user question...")
            handle_userinput(conversation, args.question)
            save_state(conversation, chat_history, raw_text, filenames, temp_state_file)
            print("Question handled and state saved.")
        else:
            print("No conversation loaded or processed. Please load a state or process documents first.")

    # Save the current state
    if args.save:
        if conversation:
            print("Saving current state...")
            save_state(conversation, chat_history, raw_text, filenames, args.save)
            print("State saved.")
        else:
            print("No conversation to save. Please load a state or process documents first.")

if __name__ == "__main__":
    main()
