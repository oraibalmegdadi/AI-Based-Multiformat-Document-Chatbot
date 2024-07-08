# Usage Examples for OfflineRAGapp-CommandLine.py

## Start a New Session with a uploading a document
* To start a new session and process the document `dams1.pdf`, use the following command:

    ```sh
    python OfflineRAGapp-CommandLine.py --docs dams1.pdf
    ```

## You Can Add Multiple Documents
* To start a new session and process multiple documents (`dams1.pdf` and `dams2.pdf`), use the following command:

    ```sh
    python OfflineRAGapp-CommandLine.py --docs dams1.pdf dams2.pdf
    ```

## Add More Documents to an Existing Session
* To add more documents (e.g., `dams3.pdf`) to the current session, use the following command:

    ```sh
    python OfflineRAGapp-CommandLine.py --add dams3.pdf
    ```

## Ask a Question
* To ask a question based on the documents processed in the current session, use the following command:

    ```sh
    python OfflineRAGapp-CommandLine.py --question "What is the purpose of a dam?"
    ```

## Save the Current Session State
* To save the current session state to a file (e.g., `session.pkl`), use the following command:

    ```sh
    python OfflineRAGapp-CommandLine.py --save session.pkl
    ```

## Load a Saved Session
* To load a previously saved session state from a file (e.g., `session.pkl`), use the following command:

    ```sh
    python OfflineRAGapp-CommandLine.py --load session.pkl
    ```
