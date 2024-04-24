import json
import streamlit as st
from spellchecker import SpellChecker
from sentence_transformers import SentenceTransformer, util

import stt
from evaluate_qa import get_score


def get_marks(model, user_answer, correct_answer):
    threshold = 0.75
    spell = SpellChecker()
    if user_answer is not None or user_answer != '':
        corrected_answer = spell.correction(user_answer)
        final_answer = corrected_answer if corrected_answer is not None else user_answer
        print("corrected", final_answer)
        user_embedding = model.encode(final_answer)
        correct_embedding = model.encode(correct_answer)
        similarity = util.pytorch_cos_sim(user_embedding, correct_embedding).item()
    else:
        similarity = 0
    st.session_state.answered = True
    return 1 if similarity > threshold else 0


def get_score(questions_and_answers, score):
    if "current_question_index" not in st.session_state:
        st.session_state.current_question_index = 0
    if "answered" not in st.session_state:
        st.session_state.answered = True

    if st.session_state.current_question_index < len(questions_and_answers) and st.session_state.answered is True:
        current_question = questions_and_answers[st.session_state.current_question_index]
        st.subheader(f"Question {st.session_state.current_question_index + 1}")
        st.write(current_question["question"])
        stt.text_to_speech(current_question["question"])
        user_answer = st.text_input("Your answer:")
        submit_button = st.button("Submit")
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

        if submit_button and st.session_state.answered is True:
            st.session_state.answered = False
            score += get_marks(model, user_answer, current_question["answer"])
            st.session_state.current_question_index += 1
            st.experimental_rerun()
    else:
        st.write("Congratulations! You have completed the test")
        st.subheader(f"Score: {score}")
    return score

def get_dict(json_file, num_of_questions):
    q_a = []
    with open(json_file, "r") as file:
        json_data = json.load(file)
    for item in json_data:
        for qa in item['qas']:

            question = qa['question']
            answer = qa['answers'][0]['text']
            question_answer = {"question": question, "answer": answer}
            q_a.append(question_answer)
            if len(q_a) > num_of_questions: break
        if len(q_a) >= num_of_questions:  break

    return q_a


def evaluate(file_name, num_of_questions):
    qa_dict = get_dict(file_name, num_of_questions)
    score = 0
    return get_score(qa_dict, score)


def main():
    num_of_questions = 5
    st.title("Voice Based Evaluation: Assessing Indian Law")
    evaluate('output.json', num_of_questions)

if __name__ == "__main__":
    main()
