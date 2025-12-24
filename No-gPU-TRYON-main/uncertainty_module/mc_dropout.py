import torch

def mc_dropout_inference(model, input_tensor, runs=10):
    model.train()  # IMPORTANT: enables dropout

    outputs = []
    with torch.no_grad():
        for _ in range(runs):
            outputs.append(model(input_tensor).unsqueeze(0))

    return torch.cat(outputs, dim=0)
