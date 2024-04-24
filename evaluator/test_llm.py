from simpletransformers.question_answering import QuestionAnsweringModel, QuestionAnsweringArgs
import json

to_predict = [
    {
        "context": "doer the others are not liable merely because when it was done they were intending tobe partakers with the doer in a different criminal act.s 34.4 Scope ambit and applicability.—Section 34 of the Indian Penal Code recognises the principle of vicarious liability incriminal jurisprudence. The said principle enshrined under Section 34 of the Codewould be attracted only if one or more than one accused person act conjointly in thecommission of offence with others. It is not necessary that all such persons should be",
        "qas": [
            {
                "question": "According to what provision in the Indian Penal Code does vicarious liability not exist?",
                "id": "0",
            }
        ]
    }
]


def train_model():
    with open(r"output.json", "r") as read_file:
        train = json.load(read_file)
    with open(r"output.json", "r") as read_file:
        test = json.load(read_file)


    model_type="bert"
    model_name= "bert-base-cased"

    model_args = QuestionAnsweringArgs()
    model_args.train_batch_size = 16
    model_args.evaluate_during_training = True
    model_args.n_best_size = 3
    model_args.num_train_epochs = 10


    train_args = {
        "reprocess_input_data": True,
        "overwrite_output_dir": True,
        "use_cached_eval_features": True,
        "output_dir": f"outputs/{model_type}",
        "best_model_dir": f"outputs/{model_type}/best_model",
        "evaluate_during_training": True,
        "max_seq_length": 128,
        "num_train_epochs": 10,
        "evaluate_during_training_steps": 1000,
        "wandb_project": "Question Answer Application",
        "wandb_kwargs": {"name": model_name},
        "save_model_every_epoch": False,
        "save_eval_checkpoints": False,
        "n_best_size":3,
        # "use_early_stopping": True,
        # "early_stopping_metric": "mcc",
        # "n_gpu": 2,
        # "manual_seed": 4,
        # "use_multiprocessing": False,
        "train_batch_size": 128,
        "eval_batch_size": 64,
        # "config": {
        #     "output_hidden_states": True
        # }
    }

    model = QuestionAnsweringModel(
        model_type, model_name, args=train_args, use_cuda=False
    )

    print("train", len(train))
    print("test", len(test))
    model.train_model(train, eval_data=test)

    result, texts = model.eval_model(test)

    to_predict = [
        {
            "context": "doer the others are not liable merely because when it was done they were intending tobe partakers with the doer in a different criminal act.s 34.4 Scope ambit and applicability.—Section 34 of the Indian Penal Code recognises the principle of vicarious liability incriminal jurisprudence. The said principle enshrined under Section 34 of the Codewould be attracted only if one or more than one accused person act conjointly in thecommission of offence with others. It is not necessary that all such persons should be",
            "qas": [
                {
                    "question": "According to what provision in the Indian Penal Code does vicarious liability not exist?",
                    "id": "0",
                }
            ],
        }
    ]

    answers, probabilities = model.predict(to_predict)

    print("answer", answers)
    print("probs", probabilities)


def load_model():

    model = QuestionAnsweringModel(model_type="bert", model_name="outputs/bert/best_model", use_cuda=False)
    answers, probabilities = model.predict(to_predict)
    print("answer", answers)
    answer = next((item for item in answers[0]['answer'] if item), None)
    print(answer)

def main():

    train_model()
    # load_model()

if __name__ == "__main__":
    main()
