from transformers import TrainingArguments

TRAINING_ARGS = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=10,
)

KD_FACTOR = 0.1
TEACHER_TEMPERATURE = 1

STUDENT_MODEL_NAME = "distilbert-base-uncased"
TEACHER_MODEL_NAME = "bhadresh-savani/bert-base-uncased-emotion"