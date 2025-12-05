import torch
import numpy as np
import logging
from config import BLEURT_CONFIG, DATASET_CONFIG

logger = logging.getLogger(__name__)

def evaluate_truthfulqa(model, tokenizer, dataset, device=None):
    if device is None:
        device = next(model.parameters()).device
    import evaluate
    bleurt = evaluate.load("bleurt", BLEURT_CONFIG["model_name"])
    # Use CPU for generation if device is MPS to avoid compatibility issues
    gen_device = torch.device("cpu") if device.type == "mps" else device
    # Move model to generation device once if needed
    model_moved = False
    if device.type == "mps":
        model.to(gen_device)
        model_moved = True
    try:
        max_score_arr = []
        acc_score_arr = []
        for example in dataset:
            question = example.get("question", "")
            correct_answers = example.get("correct_answers", [])
            incorrect_answers = example.get("incorrect_answers", [])
            if not question or not correct_answers or not incorrect_answers:
                continue
            prompt = DATASET_CONFIG["format_template"].format(question=question, best_answer="").split("Answer:")[0] + "Answer:"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(gen_device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=BLEURT_CONFIG["max_new_tokens"], do_sample=False, pad_token_id=tokenizer.eos_token_id if hasattr(tokenizer, "eos_token_id") else tokenizer.pad_token_id)
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "Answer:" in generated:
                pred = generated.split("Answer:")[-1].strip()
            else:
                pred = generated[len(prompt):].strip()
            if not pred:
                continue
            predictions_true = [pred] * len(correct_answers)
            predictions_false = [pred] * len(incorrect_answers)
            score_true = bleurt.compute(predictions=predictions_true, references=correct_answers)["scores"]
            score_false = bleurt.compute(predictions=predictions_false, references=incorrect_answers)["scores"]
            max_score = max(score_true) if score_true else 0.0
            acc_score = int(max(score_true) > max(score_false)) if score_true and score_false else 0
            max_score_arr.append(max_score)
            acc_score_arr.append(acc_score)
        avg_max_score = np.mean(max_score_arr) if max_score_arr else 0.0
        accuracy = np.mean(acc_score_arr) if acc_score_arr else 0.0
        return {"bleurt_max_score": avg_max_score, "bleurt_accuracy": accuracy}
    finally:
        # Move model back to original device if we moved it
        if model_moved:
            model.to(device)

def evaluate_qmsum(model, tokenizer, dataset, device=None, max_new_tokens=200):
    if device is None:
        device = next(model.parameters()).device
    import evaluate
    rouge = evaluate.load("rouge")
    # Use CPU for generation if device is MPS to avoid compatibility issues
    gen_device = torch.device("cpu") if device.type == "mps" else device
    # Move model to generation device once if needed
    model_moved = False
    if device.type == "mps":
        model.to(gen_device)
        model_moved = True
    try:
        predictions = []
        references = []
        for example in dataset:
            context = example.get("context", "")
            input_text = example.get("input", "")
            answer = example.get("answer", "")
            if not answer:
                continue
            prompt = f"{context}\n\n{input_text}" if context and input_text else (input_text or context)
            prompt = f"{prompt}\n\nSummary:"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(gen_device) for k, v in inputs.items()}
            input_ids_len = inputs['input_ids'].shape[1]
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id if hasattr(tokenizer, "eos_token_id") else tokenizer.pad_token_id)
            generated_ids = outputs[0][input_ids_len:]
            pred = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            if pred and answer:
                predictions.append(pred)
                references.append(answer)
        if not predictions:
            return {"rougeL": 0.0}
        result = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
        rougeL = result.get("rougeL", 0.0)
        return {"rougeL": rougeL}
    finally:
        # Move model back to original device if we moved it
        if model_moved:
            model.to(device)

def evaluate_all_datasets(model, tokenizer, datasets, device=None, qmsum_max_new_tokens=200):
    results = {}
    for dataset_name, dataset in datasets.items():
        if dataset_name == "truthfulqa":
            results[dataset_name] = evaluate_truthfulqa(model, tokenizer, dataset, device)
        elif dataset_name in ["longbench", "qmsum"]:
            results[dataset_name] = evaluate_qmsum(model, tokenizer, dataset, device, qmsum_max_new_tokens)
    return results

