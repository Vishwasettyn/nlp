from time import sleep

from spellchecker import SpellChecker
import pandas as pd
import random
from sentence_transformers import SentenceTransformer, util
import streamlit as st
import stt


def process_answer():

    sample_rate = 44100  # 44.1 kHz
    channels = 1  # Stereo
    recording_button = st.checkbox(key=random.random(), label="Click to answer", value=True)

    duration = st.slider(key=random.random(), label="Select answer duration (seconds):", min_value=10, step=5)
    if recording_button:
        audio_data = stt.record_audio(sample_rate, channels, duration)
        temp_filename = "temp.mp3"
        stt.save_audio_to_mp3(temp_filename, audio_data, sample_rate)
        answer = stt.speech_to_text(temp_filename)

    return answer


def ask_question(question, answer, model):
    st.subheader(f"Question: ", question)
    stt.text_to_speech(question)
    user_answer = process_answer()
    sleep(2)
    st.write("Answer is", user_answer)

    threshold = 0.75
    spell = SpellChecker()
    if user_answer is not None or user_answer != '':
        corrected_answer = spell.correction(user_answer)
        final_answer = corrected_answer if corrected_answer is not None else user_answer
        user_embedding = model.encode(final_answer)
        correct_embedding = model.encode(answer)
        similarity = util.pytorch_cos_sim(user_embedding, correct_embedding).item()
    else:
        similarity = 0

    return 1 if similarity > threshold else 0


def get_marks(model, user_answer, correct_answer):
    threshold = 0.75
    spell = SpellChecker()
    if user_answer is not None or user_answer != '':
        corrected_answer = spell.correction(user_answer)
        final_answer = corrected_answer if corrected_answer is not None else user_answer
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
        user_answer = process_answer()
        submit_button = st.button("Submit")
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        if submit_button and st.session_state.answered is True:
            st.session_state.answered = False
            score += get_marks(model, user_answer, current_question["answer"])
            st.session_state.current_question_index += 1
            st.rerun()
    else:
        st.write("Congratulations! You have completed the test")
        st.subheader(f"Score: {score}")
    return score


def get_dict(questions, answers, num_of_questions):
    combined = list(zip(questions, answers))
    random.shuffle(combined)
    questions, correct_answers = zip(*combined)
    qa = []
    for i in range(min(num_of_questions, len(questions))):
        question = questions[i]
        correct_answer = correct_answers[i]
        question_answer = {"question": question, "answer": correct_answer}
        qa.append(question_answer)
        if len(qa) > num_of_questions: break
    return qa


def evaluate(file_name, num_of_questions):
    df = pd.read_csv(file_name, encoding='utf-8')
    questions = df['question'].tolist()
    correct_answers = df['answer'].tolist()
    qa_dict = get_dict(questions, correct_answers, num_of_questions)
    score = 0
    return get_score(qa_dict, score)


def main():
    num_of_questions = 5
    st.title("Voice Based Evaluation: Assessing Indian Law")
    evaluate('QA_set.csv', num_of_questions)

if __name__ == "__main__":
    main()
