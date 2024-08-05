import argparse
import os
import sys

# Add the path to the parent directory to access ChatbotFunctions.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ChatbotFunctions import (
    process_documents,
    handle_userinput,
    save_state,
    load_state,
    temp_state_file,
    check_service_availability
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
    parser.add_argument("--clear", action="store_true", help="Clear the temporary state before starting")

    args = parser.parse_args()

    # Clear temporary state variables if --clear flag is used
    if args.clear:
        if os.path.exists(temp_state_file):
            os.remove(temp_state_file)
            print("Temporary state cleared.")

    conversation = None
    chat_history = []
    raw_text = ""
    filenames = []

    # Load state from a specified file if --load flag is used and no new documents are provided
    if args.load and not args.docs and not args.add:
        state = load_state(args.load, question_mode=bool(args.question), loading_session=True)
        if state:
            conversation = state['conversation']
            chat_history = state['chat_history']
            raw_text = state['raw_text']
            filenames = state.get('filenames', [])
            print(f"Pickle file '{args.load}' loaded successfully and contains analysis of the following documents:")
            for doc in filenames:
                print(f"- {doc}")
            if chat_history:
                print("Previous conversation history:")
                for i, message in enumerate(chat_history):
                    if i % 2 == 0:
                        print(f"User: {message['content']}")
                    else:
                        print(f"Bot: {message['content']}")
            return

    # Load temporary state if it exists and no specific load file is provided, and --clear flag is not set
    if not conversation and os.path.exists(temp_state_file) and not args.clear and not args.docs and not args.add:
        state = load_state(temp_state_file, question_mode=bool(args.question))
        if state:
            conversation = state['conversation']
            chat_history = state['chat_history']
            raw_text = state['raw_text']
            filenames = state.get('filenames', [])

    # Process new documents if --docs flag is used
    if args.docs:
        if check_service_availability("http://localhost:11434"):
            print("Service is available. Processing documents...")
            try:
                conversation, raw_text, new_filenames = process_documents(args.docs)
                if not new_filenames:
                    print("No valid documents provided. Exiting.")
                    return
                filenames = new_filenames
                save_state(conversation, chat_history, raw_text, filenames, temp_state_file)
                print("New session created. Current session contains analysis of the following documents:")
                for doc in filenames:
                    print(f"- {doc}")
                print("Please start asking questions to get answers based on these documents.")
            except ValueError as e:
                print(e)
                return
        else:
            print("Exiting script due to service unavailability.")
            return

    # Add documents to the existing session if --add flag is used
    if args.add:
        if check_service_availability("http://localhost:11434"):
            print("Service is available.")
            try:
                state = load_state(temp_state_file, question_mode=bool(args.question))
                if state:
                    conversation = state['conversation']
                    chat_history = state['chat_history']
                    raw_text = state['raw_text']
                    filenames = state.get('filenames', [])
                else:
                    print("No existing session found to add documents to. Exiting.")
                    return

                existing_docs = []
                missing_docs = []
                for doc in args.add:
                    if os.path.exists(doc):
                        existing_docs.append(doc)
                    else:
                        missing_docs.append(doc)

                if missing_docs:
                    print("The following documents were not found:")
                    for doc in missing_docs:
                        print(f"- {doc}")
                    if not existing_docs:
                        print("No valid documents provided. Exiting.")
                        return

                conversation, raw_text, new_filenames = process_documents(existing_docs, raw_text)
                if not new_filenames:
                    print("No valid text extracted from the documents. Exiting.")
                    return

                filenames.extend(new_filenames)
                save_state(conversation, chat_history, raw_text, filenames, temp_state_file)
                print("Documents added successfully. Current session contains analysis of the following documents:")
                for doc in filenames:
                    print(f"- {doc}")
            except ValueError as e:
                print(e)
                return
        else:
            print("Exiting script due to service unavailability.")
            return

    # Handle user's question if --question flag is used
    if args.question:
        if conversation:
            response = handle_userinput(conversation, args.question)
            if not response:
                response = "I couldn't find an answer to your question."
            save_state(conversation, chat_history, raw_text, filenames, temp_state_file)
            print(f"User: {args.question}")
            print(f"Bot: {response}")
        else:
            print("No conversation loaded or processed. Please load a state or process documents first.")

    # Save the current state if --save flag is used
    if args.save:
        if conversation:
            save_state(conversation, chat_history, raw_text, filenames, args.save)
            print("Session saved successfully.")
        else:
            print("No conversation to save. Please load a state or process documents first.")

if __name__ == "__main__":
    main()
