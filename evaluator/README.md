# Voice based Evaluator

Voice based Evaluator is an LLM based project aims at evaluating the expertise of a student in particular domain. (Indian Penal code in our case)

### The project is executed as multiple stages 
1. Question generation
2. Answer Generation
3. Posting questions to student
4. Retrieve answer from student
5. Evauluate the answer and provide score


### 1. Question Generation
In the question generation workflow, the initial step centers around breaking down a lengthy text into smaller, digestible portions or document chunks. Each of these chunks is then subjected to a summarization process guided by a predefined PromptTemplate. The base model that is emplyed here is Mistral-7B-Instruct-v0.1
The second stage involves leveraging the extracted questions from individual document chunks to construct a more expansive and comprehensive list of questions.
![image](https://github.com/Vishwasettyn/nlp/assets/26715081/b2043127-e322-4935-ab4a-b7b2f0ffb440)

### 2. Answer Generation
The process of generating answers involves breaking down the input text into smaller document chunks, each of which is summarized, and embeddings are created to represent them effectively. These embeddings are then stored in a vector database. By using the Google Flan model, the system retrieves the most similar document when a question is posed, enabling the generation of precise answers.
This iterative approach ensures that each question is systematically processed through the summarized document chunks, and relevant answer is extracted, providing accurate and contextually appropriate answers.

![image](https://github.com/Vishwasettyn/nlp/assets/26715081/41a06a69-5192-48c2-b0a0-c9523ee0d678)

### 3.Questioning
Streamlit is being used for creation of web based application. 
Python's ‘sounddevice’ library is used for managing real-time audio input/output.
Pydub is used to manage audio files through an easy-to-use interface.
Google Text-to-Speech (gTTS) is a powerful library that converts text questions to speech.

### 4.Answering
Streamlit is being used for creation of web based application. 
Whisper is an automatic speech recognition (ASR) mdoel for Speech to text
Sentence Transformers and Vectorisation is used to covert user provided answers to vectors 

### 5.Evaluation and Scoring
Method1: flan-t5-xxl model is used for vectorisation and Cosine similarity is used for similarity  check with correct answers in vector db vs user provided answers from (4)
Method 2: bert-base-cased model is used for training data set using SimpleTransformer to generate question-answer trained model on top of which questions are posted to get answers. WANDB is used to evaluate training data set.

![image](https://github.com/Vishwasettyn/nlp/assets/26715081/e12cd620-acb4-4360-a56a-2a312ce76f18)





![image](https://github.com/Vishwasettyn/nlp/assets/26715081/1bc68c55-96a2-4dac-8d6c-b96badc5f6a5)
![Uploading image.png…]()

