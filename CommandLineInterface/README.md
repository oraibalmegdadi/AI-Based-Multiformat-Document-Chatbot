# Install requirements 


Step 1:Clone the repository from GitHub:
```sh
git clone https://github.com/oraibalmegdadi/AI-Based-Multiformat-Document-Chatbot
```
Navigate to your folder: 

cd yourprojectname

Step 2: Create a Conda Environment
Create a New Conda Environment:
```sh
conda create --name newenv python=3.8
```

Activate the Conda Environment:

```sh
conda activate newenv
```

Step 2: Install Required Packages
Install the packages listed in your requirements.txt file within the new conda environment. Note that some packages might be available through conda directly, while others might need to be installed via pip.

Install Common Packages via Conda:

```sh
conda install faiss-cpu pytorch torchaudio torchvision -c pytorch
```

Install Remaining Packages via Pip:

```sh
pip install -r ChatbotCL-Requirements.txt

```


# Usage Examples for ChatbotCL.py


##Make sure to activate the Conda Environment which is created previously:

```sh
conda activate newenv
```


## Start a New Session:
* To start a new session and process one document `dams1.pdf`, use the following command:

    ```sh
    python ChatbotCL.py --docs dams1.pdf
    ```
* To start a new session and process multiple documents (`dams1.pdf` and `dams2.pdf`), use the following command:

    ```sh
    python ChatbotCL.py --docs dams1.pdf dams2.pdf
    ```

## Add More Documents to an Existing Session
* To add more documents (e.g., `dams3.pdf`) to the current session, use the following command:

    ```sh
    python ChatbotCL.py --add dams3.pdf
    ```

## Ask a Question
* To ask a question based on the documents processed in the current session, use the following command:

    ```sh
    python ChatbotCL.py --question "What is the purpose of a dam?"
    ```

## Save the Current Session State
* To save the current session state to a file (e.g., `session.pkl`), use the following command:

    ```sh
    python ChatbotCL.py --save session.pkl
    ```

## Load a Saved Session
* To load a previously saved session state from a file (e.g., `session.pkl`), use the following command:

    ```sh
    python ChatbotCL.py --load session.pkl
    ```
