TRAINING_ARGUMENTS = {
    "eval_strategy": "epoch",
    "learning_rate": 2e-5,
    "per_device_train_batch_size": 64,
    "per_device_eval_batch_size": 64,
    "num_train_epochs": 3,
    "weight_decay": 0.01,
    "logging_steps": 10,
    "fp16": True,
    "dataloader_num_workers": 2,
    "dataloader_pin_memory": True,
}

OUTPUT_DIR = "./results"
PRUNING_SPARSITY = 0.2

KD_FACTOR = 0.1
TEACHER_TEMPERATURE = 1

STUDENT_MODEL_NAME = "distilbert-base-uncased"
TEACHER_MODEL_NAME = "bhadresh-savani/bert-base-uncased-emotion"