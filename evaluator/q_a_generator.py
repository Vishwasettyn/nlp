import json
from http.client import HTTPException
from time import sleep
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import CTransformers
import os
from PyPDF2 import PdfReader
import csv
import re
from langchain.chains import LLMChain


def clean_doc(text):
    pattern = re.compile(r'\n\d+.*?(\n|$)', re.DOTALL)  # remove unnecessary lines
    cleaned_text = re.sub(pattern, '\n', text)
    return cleaned_text


def get_pdf_text(file_path):
    pdf_docs = open(os.path.join(file_path), 'rb')
    pdf_reader = PdfReader(pdf_docs)
    text = ''
    for page in pdf_reader.pages:
        text += re.sub(r'[,]', '', page.extract_text().replace(str, ""))
        pattern = re.compile(r'^\d+\.\s*.*?(?=(\n\[s ([\d.]+)\]))', re.MULTILINE | re.DOTALL)
        text = re.sub(pattern, '', text)

    return text


def file_processing(file_path):
    question_gen = get_pdf_text(file_path)
    splitter_ques_gen = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100
    )

    chunks_ques_gen = splitter_ques_gen.split_text(question_gen)
    document_ques_gen = [Document(page_content=t) for t in chunks_ques_gen]
    splitter_ans_gen = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    document_answer_gen = splitter_ans_gen.split_documents(
        document_ques_gen
    )

    return document_ques_gen, document_answer_gen


def find_index(sentence1, sentence2):
    # to find the index where the answer lies in the chunk
    index = sentence1.find(sentence2)
    if index != -1:
        return index

    words = sentence2.split()
    for word in words:
        if word in sentence1:
            return sentence1.index(word)
    return -1


def get_json_dataset(doc, question, answer, id):
    index = find_index(doc, answer)
    if index == -1:
        is_impossible = True
        answer = ""
    else:
        is_impossible = False
    data = {
        "context": doc,
        "qas": [
            {
                "id": f"{id}",
                "is_impossible": is_impossible,
                "question": question,
                "answers": [
                    {
                        "text": answer,
                        "answer_start": index
                    }
                ]
            }
        ]
    }
    return data


def get_answer(query, vector_store, chain):
    docs = vector_store.similarity_search(query)
    doc = (docs[0])
    answers = chain.run(input_documents=docs, question=query)
    return doc.page_content, answers


def answer_generator(document_answer_gen):

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    vector_store = FAISS.from_documents(document_answer_gen, embeddings)

    llm_answer_gen = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.5, "max_length": 512})

    chain = load_qa_chain(llm_answer_gen, chain_type="stuff")
    return vector_store, chain


def question_answer_generator(file_path, output_file):

    document_ques_gen, document_answer_gen = file_processing(file_path)

    llm_ques_gen_pipeline = CTransformers(
        model="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
        model_type="mistral",
        max_new_tokens=1048,
        temperature=0.4
    )

    prompt_template = """
       You are an expert at creating questions based on summarized data.
       Your goal is to prepare a set of questions using the information below.

       ------------
       {text}
       ------------

       QUESTIONS:
       """

    question_prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

    ques_gen_chain = LLMChain(llm=llm_ques_gen_pipeline, prompt=question_prompt)

    vector_store, answer_chain = answer_generator(document_answer_gen)

    questions = []
    dataset = []
    ques_list = []
    counter = 0
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["question", "answer"])
        for docs in document_ques_gen:
            try:
                ques = ques_gen_chain.run(docs)
                quests = ques.split('\n')

                # questions = [sentence.split('. ', 1)[1] if '. ' in sentence else sentence for sentence in quests if sentence]
                questions = [sentence.replace('\t', '').split('. ', 1)[1] if '. ' in sentence else sentence.replace('\t', '')
                             for sentence in quests if sentence.replace('\t', '')]
                ques_list.extend(questions)
            except Exception as e:
                print(f"An exception occurred while question generation: {e}. Skipping this iteration.")

            filtered_ques_list = [element for element in questions if element.endswith('?') or element.endswith('.')]

            i = len(dataset)
            for question in filtered_ques_list:
                try:
                    print("Question: ", question)
                    doc, answers = get_answer(question, vector_store, answer_chain)
                    # remove all special characters from sentences
                    answer = re.sub(r'[\n{}[\]()]', '', answers)
                    question = re.sub(r'[\n{}[\]()]', '', question)
                    doc = re.sub(r'[\n{}[\]()]', '', doc)
                    print("Answer: ", answer)
                    print("--------------------------------------------------\n\n")
                    csv_writer.writerow([question, answer])
                    i = i+1
                    dataset.append(get_json_dataset(doc, question, answer, i))
                    with open("output.json", "w") as json_file:
                        json.dump(dataset, json_file, indent=2)

                except HTTPException as e:
                    if "status code 429" in str(e):
                        print("Too many requests. Sleeping for one hour as limit reached 300.")
                        sleep((60 * 60) + 10)
                    else:
                        print(f"HTTPException: {e}")

                except Exception as e:
                    print(f"An exception occurred while answer generation: {e}. Skipping this iteration.")

    return output_file


def get_qa_set(file_path, output_path):

    base_folder = './'
    if not os.path.isdir(base_folder):
        os.mkdir(base_folder)
    output_file = base_folder + output_path
    return question_answer_generator(file_path, output_file)


def main():
    load_dotenv()
    get_qa_set("./IPC.pdf", "QA_set.csv")
    print("QA set created")


if __name__ == "__main__":
    main()
