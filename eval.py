from tqdm import tqdm
from models.VQA_Model import VQAModel
from metrics import exact_match, bleu
from dataset import VQADataset
import torch
import csv
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

def evaluate_model(model, dataset, device):
    results = []
    total_items = len(dataset)
    exact_match_score = 0
    bleu_score = 0

    pbar = tqdm(dataset, desc="Evaluating", unit="item")
    for idx, item in enumerate(pbar):
        image_path = item[0]
        question = item[1]
        answers = item[2]

        pred = model.predict(image_path, question)
        em = exact_match(pred, answers)
        bl = bleu(pred, answers)

        exact_match_score += em
        bleu_score += bl
        image_path_csv = f"img{idx+1}.jpg" 
        results.append([image_path_csv, question, answers, pred, em, bl])
        pbar.set_description(f"bleu_score: {bleu_score/total_items:.4f}, exact_match_score: {exact_match_score/total_items:.4f}")

    # Final scores
    final_exact_match = exact_match_score / total_items
    final_bleu = bleu_score / total_items
    print("Final Exact Match Score:", final_exact_match)
    print("Final BLEU Score:", final_bleu)

    # Save CSV
    with open("vqa_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "question", "answers", "prediction", "EM", "BLEU"])
        writer.writerows(results)

    print("Saved detailed results to vqa_results.csv")
  
        
dataset = VQADataset("data/images", "data/questions.json")
model = VQAModel(device=device)
evaluate_model(model, dataset, device)