# %%
import torch
import sys, os
from pruning import pruning_functions
from constants import *
from torch.nn import functional as F
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

def get_training_args(**overrides) -> TrainingArguments:
    arguments = {**TRAINING_ARGUMENTS, **overrides}
    return TrainingArguments(**arguments)

# %%
class CustomTrainer(Trainer):
    def __init__(self, teacher=None, temperature=TEACHER_TEMPERATURE, kd_factor=KD_FACTOR, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if teacher:
            for param in teacher.parameters():
                param.requires_grad = False
            teacher.eval()
            teacher = teacher.to(self.args.device)
        self.teacher = teacher
        self.temperature = temperature
        self.kd_factor = kd_factor

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        logits = outputs.logits

        # Supervised loss
        supervised_loss = F.cross_entropy(logits, inputs.get("labels"))
        if not self.teacher:
            return (supervised_loss, outputs) if return_outputs else supervised_loss

        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs)
            teacher_logits = teacher_outputs.logits

        student_probs = F.log_softmax(logits/self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits/self.temperature, dim=-1)

        # KD loss
        kd_loss = F.kl_div(student_probs, teacher_probs, log_target=False, reduction="batchmean")*self.temperature**2


        # Combined
        loss =  (1-self.kd_factor)*supervised_loss + self.kd_factor*kd_loss

        return (loss, outputs) if return_outputs else loss

def train_model(strategy=None, teacher=None):
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_NAME)
    
    train_dataset = load_dataset("dair-ai/emotion", split="train")
    val_dataset = load_dataset("dair-ai/emotion", split="validation")

    tokenized_train = train_dataset.map(lambda x: tokenizer(x["text"], truncation=True), batched = True)
    tokenized_val = val_dataset.map(lambda x: tokenizer(x["text"], truncation=True), batched = True)

    num_labels = train_dataset.features["label"].num_classes
    student = AutoModelForSequenceClassification.from_pretrained(STUDENT_MODEL_NAME, num_labels=num_labels,dtype=torch.float32)
    training_args = get_training_args(output_dir= os.path.join(OUTPUT_DIR, strategy))
    trainer = CustomTrainer(
        teacher=teacher,
        model=student,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
    )
    trainer.train()
    final_model_path = os.path.join(OUTPUT_DIR, strategy, "final")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)


if __name__=='__main__':
    teacher = AutoModelForSequenceClassification.from_pretrained(TEACHER_MODEL_NAME, dtype=torch.float32)
    for name in pruning_functions:
        func = pruning_functions[name]
        pruned_teacher = func(teacher, PRUNING_SPARSITY)
        train_model(name, pruned_teacher)

    # train student without kd
    train_model("Without KD", None)