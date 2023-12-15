import clip
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(torch.cuda.is_available())
model, transform = clip.load("ViT-B/32", device=device)
model.to(device)

def prompt_encode(prompt, device="cuda"):
    """
    Args:
        prompt (str): the text prefix before the class

    Returns:
        text_inputs(torch.Tensor)

    """
    text = clip.tokenize(prompt).to(device)
    print("Transform text into token:",text.size())
    print(text)
    text_inputs = model.encode_text(text)

    return text_inputs

# Example
prompt_text = "a photo of a"
encoded_text = prompt_encode(prompt_text)
print("Transform token into code:",encoded_text.size())
print(encoded_text)
