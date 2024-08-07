# Command Line Version: Installation Steps and Examples

## Installation Requirements

### Step 1: Clone the Repository from GitHub
Clone the repository to your local machine using the following command:

```sh
git clone https://github.com/oraibalmegdadi/AI-Based-Multiformat-Document-Chatbot
```

Navigate to the project folder:

```sh
cd /the project path
```

### Step 2: Create a Conda Environment
Create a new Conda environment with Python 3.8:

```sh
conda create --name newenv python=3.8
```

Activate the Conda environment:


```sh
conda activate newenv
```

### Step 3: Install Required Packages
Install the packages listed in your requirements.txt file within the new Conda environment. Note that some packages might be available through Conda directly, while others might need to be installed via pip.

Install common packages via Conda:
```sh
conda install faiss-cpu pytorch torchaudio torchvision -c pytorch
```

Install remaining packages via pip:


```sh
pip install -r ChatbotCL-Requirements.txt

```



## Usage Examples for `ChatbotCL.py`

### Activate the Conda Environment
Ensure the Conda environment is activated:

```sh
conda activate newenv
```

### Start a New Session:
* To start a new session and process one document `dams1.pdf`, use the following command:

    ```sh
    python ChatbotCL.py --docs dams1.pdf
    ```
* To start a new session and process multiple documents (`dams1.pdf` and `dams2.pdf`), use the following command:

    ```sh
    python ChatbotCL.py --docs dams1.pdf dams2.pdf
    ```

### Add More Documents to an Existing Session
* To add more documents (e.g., `dams3.pdf`) to the current session, use the following command:

    ```sh
    python ChatbotCL.py --add dams3.pdf
    ```

### Ask a Question
* To ask a question based on the documents processed in the current session, use the following command:

    ```sh
    python ChatbotCL.py --question "What is the purpose of a dam?"
    ```

### Save the Current Session State
* To save the current session -which contains the analysis of existing documents without needing to upload and analyze them again- to a file (e.g., `session.pkl`), use the following command:

    ```sh
    python ChatbotCL.py --save session.pkl
    ```

### Load a Saved Session
* To load a previously saved session state from a file (e.g., `session.pkl`), use the following command:

    ```sh
    python ChatbotCL.py --load session.pkl
    ```
