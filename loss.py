import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class CustomLoss(torch.nn.Module):
    def __init__(self, teacher=None, temperature=1, kd_factor=0):
        super().__init__()
        if not teacher:
            model_name = "bhadresh-savani/bert-base-uncased-emotion"
            teacher = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=torch.float32)
        for param in teacher.parameters():
            param.requires_grad = False
        self.teacher = teacher
        self.temperature = temperature
        self.kd_factor = kd_factor

    def forward(self, model, inputs):
        outputs = model(**inputs)
        logits = outputs.logits

        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs)
            teacher_logits = teacher_outputs.logits
            teacher_logits /= self.temperature
            teacher_probs = F.softmax(teacher_logits)

        # KD loss
        kd_loss = F.cross_entropy(logits, teacher_probs)

        # Supervised loss
        supervised_loss = F.cross_entropy(logits, inputs["labels"])

        return (1-self.kd_factor)*supervised_loss + self.kd_factor*kd_loss

if __name__=='__main__':
    model_name = "bhadresh-savani/bert-base-uncased-emotion"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer("Hello", return_tensors="pt")
    inputs["labels"] = torch.tensor([0])
    loss = CustomLoss()
    print(loss(model, inputs))