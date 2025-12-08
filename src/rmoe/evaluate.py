import torch
import numpy as np
import logging
import gc
from config import BLEURT_CONFIG, DATASET_CONFIG

logger = logging.getLogger(__name__)

def evaluate_truthfulqa(model, tokenizer, dataset, device=None, temperature=0.0):
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
        all_predictions = []
        all_correct_refs = []
        all_incorrect_refs = []
        
        logger.info(f"Starting TruthfulQA evaluation on {len(dataset)} examples...")
        
        for idx, example in enumerate(dataset):
            question = example.get("question", "")
            correct_answers = example.get("correct_answers", [])
            incorrect_answers = example.get("incorrect_answers", [])
            if not question or not correct_answers or not incorrect_answers:
                continue
            prompt = DATASET_CONFIG["format_template"].format(question=question, best_answer="").split("Answer:")[0] + "Answer:"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(gen_device) for k, v in inputs.items()}
            with torch.no_grad():
                do_sample = temperature > 0.0
                pad_token_id = tokenizer.eos_token_id if hasattr(tokenizer, "eos_token_id") else tokenizer.pad_token_id
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=BLEURT_CONFIG["max_new_tokens"], 
                    temperature=temperature if do_sample else None, 
                    do_sample=do_sample, 
                    pad_token_id=pad_token_id,
                    repetition_penalty=BLEURT_CONFIG.get("repetition_penalty", 1.0),
                    use_cache=True
                )
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            pred = generated.split("Answer:")[-1].strip() if "Answer:" in generated else generated[len(prompt):].strip()
            
            if idx < 5:
                logger.info(f"\n{'='*80}")
                logger.info(f"Example {idx + 1}/{len(dataset)}")
                logger.info(f"Input Prompt: {prompt}")
                logger.info(f"Generated Answer: {pred}")
                logger.info(f"Correct Answers (sample): {correct_answers[0] if correct_answers else 'N/A'}")
                logger.info(f"Incorrect Answers (sample): {incorrect_answers[0] if incorrect_answers else 'N/A'}")
                logger.info(f"{'='*80}\n")
            
            if pred:
                all_predictions.append(pred)
                all_correct_refs.append(correct_answers)
                all_incorrect_refs.append(incorrect_answers)
        
        if not all_predictions:
            return {"bleurt_max_score": 0.0, "bleurt_accuracy": 0.0}
        
        max_score_arr = []
        acc_score_arr = []
        batch_size = 32
        
        for i in range(0, len(all_predictions), batch_size):
            batch_preds = all_predictions[i:i+batch_size]
            batch_correct = all_correct_refs[i:i+batch_size]
            batch_incorrect = all_incorrect_refs[i:i+batch_size]
            
            expanded_preds_true = [p for p, refs in zip(batch_preds, batch_correct) for _ in refs]
            expanded_refs_true = [r for refs in batch_correct for r in refs]
            expanded_preds_false = [p for p, refs in zip(batch_preds, batch_incorrect) for _ in refs]
            expanded_refs_false = [r for refs in batch_incorrect for r in refs]
            
            scores_true = bleurt.compute(predictions=expanded_preds_true, references=expanded_refs_true)["scores"] if expanded_preds_true else []
            scores_false = bleurt.compute(predictions=expanded_preds_false, references=expanded_refs_false)["scores"] if expanded_preds_false else []
            
            true_idx = 0
            false_idx = 0
            for correct_refs, incorrect_refs in zip(batch_correct, batch_incorrect):
                example_scores_true = scores_true[true_idx:true_idx+len(correct_refs)]
                example_scores_false = scores_false[false_idx:false_idx+len(incorrect_refs)]
                true_idx += len(correct_refs)
                false_idx += len(incorrect_refs)
                max_score = max(example_scores_true) if example_scores_true else 0.0
                acc_score = int(max(example_scores_true) > max(example_scores_false)) if example_scores_true and example_scores_false else 0
                max_score_arr.append(max_score)
                acc_score_arr.append(acc_score)
            del batch_preds, batch_correct, batch_incorrect, expanded_preds_true, expanded_refs_true, expanded_preds_false, expanded_refs_false, scores_true, scores_false
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        avg_max_score = np.mean(max_score_arr) if max_score_arr else 0.0
        accuracy = np.mean(acc_score_arr) if acc_score_arr else 0.0
        return {"bleurt_max_score": avg_max_score, "bleurt_accuracy": accuracy}
    finally:
        if model_moved:
            model.to(device)

def evaluate_qmsum(model, tokenizer, dataset, device=None, max_new_tokens=200, temperature=0.0):
    if device is None:
        device = next(model.parameters()).device
    import evaluate
    rouge = evaluate.load("rouge")
    gen_device = torch.device("cpu") if device.type == "mps" else device
    model_moved = False
    if device.type == "mps":
        model.to(gen_device)
        model_moved = True
    try:
        predictions = []
        references = []
        for example in dataset:
            answer = example.get("answer", "")
            if not answer:
                continue
            context = example.get("context", "")
            input_text = example.get("input", "")
            prompt = f"{context}\n\n{input_text}" if context and input_text else (input_text or context)
            prompt = f"{prompt}\n\nSummary:"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=32768)
            inputs = {k: v.to(gen_device) for k, v in inputs.items()}
            input_ids_len = inputs['input_ids'].shape[1]
            with torch.no_grad():
                do_sample = temperature > 0.0
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens, 
                    temperature=temperature if do_sample else None, 
                    do_sample=do_sample, 
                    pad_token_id=tokenizer.eos_token_id if hasattr(tokenizer, "eos_token_id") else tokenizer.pad_token_id,
                    repetition_penalty=BLEURT_CONFIG.get("repetition_penalty", 1.0)
                )
            full_sequences = outputs.sequences[0] if hasattr(outputs, "sequences") else outputs[0]
            generated_ids = full_sequences[input_ids_len:]
            full_text = tokenizer.decode(full_sequences, skip_special_tokens=True).strip()
            input_decoded = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True).strip()
            pred = full_text[len(input_decoded):].strip() if full_text.startswith(input_decoded) else tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            if pred and answer:
                predictions.append(pred)
                references.append(answer)
        if not predictions:
            return {"rougeL": 0.0}
        result = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
        return {"rougeL": result.get("rougeL", 0.0)}
    finally:
        if model_moved:
            model.to(device)

def evaluate_all_datasets(model, tokenizer, datasets, device=None, qmsum_max_new_tokens=200, temperature=0.0):
    results = {}
    for dataset_name, dataset in datasets.items():
        if dataset_name == "truthfulqa":
            results[dataset_name] = evaluate_truthfulqa(model, tokenizer, dataset, device, temperature)
        elif dataset_name in ["longbench", "qmsum"]:
            results[dataset_name] = evaluate_qmsum(model, tokenizer, dataset, device, qmsum_max_new_tokens, temperature)
    return results

