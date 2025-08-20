from transformers import BlipForQuestionAnswering, BlipProcessor
import torch

class VQAModel:
    def __init__(self, device):
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base", use_fast=True)
        self.model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)
        self.device = device

    def predict(self, image, question):
        inputs = self.processor(images=image, text=question, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(**inputs)
        return self.processor.decode(output_ids[0], skip_special_tokens=True)
