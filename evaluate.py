# %%
import os
import torch
import numpy as np
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from scipy.stats import kendalltau
from scipy.spatial.distance import jensenshannon
from tabulate import tabulate

from constants import *
from pruning import pruning_functions
from pruning.magnitude_prune import sparsity_report

device = "cuda" if torch.cuda.is_available() else "cpu"
EMOTIONS = ["sadness", "joy", "love", "anger", "fear", "surprise"]


# %%
def evaluate_classification(model, tokenizer, dataset):
    model.eval()
    model.to(device)

    labels = np.array(dataset["label"])
    predictions = []
    texts = dataset["text"]
    batch_size = TRAINING_ARGUMENTS["per_device_eval_batch_size"]

    for i in range(0, len(dataset), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            batch_predictions = outputs.logits.argmax(dim=-1).cpu().numpy()
            predictions.extend(batch_predictions)

    predictions = np.asarray(predictions)

    overall_acc = accuracy_score(labels, predictions)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(labels, predictions, average="macro", zero_division=0)
    precision, recall, weighted_f1, support = precision_recall_fscore_support(labels, predictions, average="weighted", zero_division=0)

    class_p, class_r, class_f1, true_count = precision_recall_fscore_support(labels, predictions, average=None, zero_division=0)
    cm = confusion_matrix(labels, predictions)
    class_acc = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)

    return {
        "overall_acc": overall_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_p": macro_p,
        "macro_r": macro_r,
        "class_acc": class_acc,
        "class_p": class_p,
        "class_r": class_r,
        "class_f1": class_f1,
        "true_count": true_count,
    }


# %%
def evaluate_attention(model, tokenizer, dataset, num_samples=200):
    model.eval()
    model.to(device)
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation("eager")

    # Jain & Wallace (2019)
    attention_gradient_correlation = []
    divergence_score = []
    
    # Jacovi & Goldberg (2020)
    faithfulness_comprehensiveness = []
    faithfulness_sufficiency = []
    attention_sharpness_entropy = []
    
    mask_id = tokenizer.mask_token_id or tokenizer.pad_token_id

    for i in range(min(num_samples, len(dataset))):
        inputs = tokenizer(dataset[i]["text"], truncation=True, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        if input_ids.shape[1] <= 4:
            continue

        embeds = model.get_input_embeddings()(input_ids).detach()
        embeds.requires_grad_(True)

        model.zero_grad()
        outputs = model(inputs_embeds=embeds, output_attentions=True)
        probs = F.softmax(outputs.logits, dim=-1)
        pred_cls = probs.argmax(dim=-1).item()
        orig_prob = probs[0, pred_cls].item()

        outputs.logits[0, pred_cls].backward()

        attn = outputs.attentions[-1][0, :, 0, :].mean(dim=0).detach().cpu().numpy()[1:-1]
        grad_in = (embeds.grad[0] * embeds[0]).norm(p=2, dim=-1).detach().cpu().numpy()[1:-1]

        attn /= attn.sum() + 1e-12
        grad_in /= grad_in.sum() + 1e-12


        attention_gradient_correlation.append(kendalltau(attn, grad_in)[0])
        divergence_score.append(jensenshannon(attn, grad_in) ** 2)


        attention_sharpness_entropy.append(-np.sum(attn * np.log(attn + 1e-12)) / np.log(len(attn)))
        sorted_idx = np.argsort(attn)[::-1] + 1

        comp_ids = input_ids.clone()
        comp_ids[0, sorted_idx[0]] = mask_id

        k_keep = max(1, int(np.ceil(len(attn) * 0.2)))
        suff_ids = input_ids.clone()
        suff_ids[0, sorted_idx[k_keep:]] = mask_id

        with torch.no_grad():
            comp_probs = F.softmax(model(comp_ids).logits, dim=-1)
            suff_probs = F.softmax(model(suff_ids).logits, dim=-1)

        faithfulness_comprehensiveness.append(orig_prob - comp_probs[0, pred_cls].item())
        faithfulness_sufficiency.append(orig_prob - suff_probs[0, pred_cls].item())

    return {
        "attention_gradient_correlation": float(np.nanmean(attention_gradient_correlation)),
        "divergence_score": float(np.nanmean(divergence_score)),
        "faithfulness_comprehensiveness": float(np.mean(faithfulness_comprehensiveness)),
        "faithfulness_sufficiency": float(np.mean(faithfulness_sufficiency)),
        "attention_sharpness_entropy": float(np.mean(attention_sharpness_entropy)),
    }


# %%
def evaluate_model(model, tokenizer, dataset, name):
    sparsity = sparsity_report(model)["overall"]
    cls_res = evaluate_classification(model, tokenizer, dataset)
    attn_res = evaluate_attention(model, tokenizer, dataset)

    print(f"\n   {name}  \n")
    
    overall_table = [
        ["Sparsity", f"{sparsity * 100:.2f}%"],
        ["Overall Accuracy", f"{cls_res['overall_acc'] * 100:.2f}%"],
        ["Macro F1 / Weighted F1", f"{cls_res['macro_f1']:.4f} / {cls_res['weighted_f1']:.4f}"],
        ["Macro Precision / Recall", f"{cls_res['macro_p']:.4f} / {cls_res['macro_r']:.4f}"]
    ]
    print(tabulate(overall_table, headers=["Metric", "Value"], tablefmt="fancy_grid"))

    class_table = []
    for idx, emo in enumerate(EMOTIONS):
        class_table.append([
            emo,
            f"{cls_res['class_acc'][idx]*100:.2f}%",
            f"{cls_res['class_p'][idx]:.4f}",
            f"{cls_res['class_r'][idx]:.4f}",
            f"{cls_res['class_f1'][idx]:.4f}",
            cls_res['true_count'][idx]
        ])
    print("\nClass-Wise Breakdown:")
    print(tabulate(class_table, headers=["Emotion", "Accuracy", "Precision", "Recall", "F1-Score", "True Count"], tablefmt="fancy_grid"))

    attn_table = [[k, f"{v:.4f}"] for k, v in attn_res.items()]
    print("\nAttention Explainability Metrics:")
    print(tabulate(attn_table, headers=["Metric", "Score"], tablefmt="fancy_grid"))

    return {"sparsity": sparsity, **cls_res, **attn_res}


# %%
if __name__ == "__main__":
    test_dataset = load_dataset("dair-ai/emotion", split="test")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_NAME)
    teacher = AutoModelForSequenceClassification.from_pretrained(TEACHER_MODEL_NAME, dtype=torch.float32)

    evaluate_model(teacher, tokenizer, test_dataset, "Unpruned Teacher")

    for name, func in {**pruning_functions, "Without KD": None}.items():
        if func:
            pruned_teacher = func(teacher, PRUNING_SPARSITY)
            evaluate_model(pruned_teacher, tokenizer, test_dataset, f"Pruned Teacher ({name})")

        student_path = os.path.join(OUTPUT_DIR, name, "final")
        if os.path.exists(student_path):
            student = AutoModelForSequenceClassification.from_pretrained(student_path, dtype=torch.float32)
            student_tokenizer = AutoTokenizer.from_pretrained(student_path)
            evaluate_model(student, student_tokenizer, test_dataset, f"Distilled Student ({name})")
