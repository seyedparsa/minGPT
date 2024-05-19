import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F
import os, random

from model import TransformerClassifier, TransformerLM
from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset

seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers


eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set


## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input 
## size of 64, hidden size of 50 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15  # epochs for classifier training


def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data. 
    """
    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts


def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)  
    return padded_sequences, labels


def compute_classifier_accuracy(classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            outputs = classifier(X)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
        accuracy = (100 * total_correct / total_samples)
        classifier.train()
        return accuracy


def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses = []
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        loss, logits = decoderLMmodel(X, Y)  # your model should be computing the cross entropy loss
        losses.append(loss.item())
        # total_loss += loss.item()
        if len(losses) >= eval_iters:
            break

    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity


def classification(tokenizer, data_folder):
    train_CLS_dataset = SpeechesClassificationDataset(tokenizer, data_folder + "/train_CLS.tsv")
    train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=True)
    test_CLS_dataset = SpeechesClassificationDataset(tokenizer, data_folder + "/test_CLS.tsv")
    test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch)

    # train a classifier transformer for #epochs_CLS epochs with AdamW optimizer and crossentropy loss:
    cls_model = TransformerClassifier(tokenizer.vocab_size, block_size, n_embd, n_layer, n_head, n_hidden, n_output)
    optimizer = torch.optim.AdamW(cls_model.parameters(), lr=learning_rate)
    for epoch in range(epochs_CLS):
        train_loss = 0
        for xb, yb in train_CLS_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = cls_model(xb)
            loss = F.cross_entropy(logits, yb)
            train_loss += loss
            loss.backward()
            optimizer.step()

        train_accuracy = compute_classifier_accuracy(cls_model, train_CLS_loader)
        test_accuracy = compute_classifier_accuracy(cls_model, test_CLS_loader)

        print(f"Epoch [{epoch + 1}/{epochs_CLS}], "
              f"Train Loss: {train_loss / len(train_CLS_loader):.4f}, "
              f"Train Accuracy: {train_accuracy:.2f}%, "
              f"Test Accuracy: {test_accuracy:.2f}%")


def language_modeling(tokenizer, data_folder):
    train_LM_dataset = LanguageModelingDataset(tokenizer, data_folder + "/train_LM.txt", block_size)
    train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)

    test_LM_hbush_dataset = LanguageModelingDataset(tokenizer, data_folder + "/test_LM_hbush.txt", block_size)
    test_LM_obama_dataset = LanguageModelingDataset(tokenizer, data_folder + "/test_LM_obama.txt", block_size)
    test_LM_wbush_dataset = LanguageModelingDataset(tokenizer, data_folder + "/test_LM_wbush.txt", block_size)
    test_LM_hbush_loader = DataLoader(test_LM_hbush_dataset, batch_size=batch_size)
    test_LM_obama_loader = DataLoader(test_LM_obama_dataset, batch_size=batch_size)
    test_LM_wbush_loader = DataLoader(test_LM_wbush_dataset, batch_size=batch_size)

    # iterate over the training data for a fixed number of iterations:
    language_model = TransformerLM(tokenizer.vocab_size, block_size, n_embd, n_layer, n_head)
    optimizer = torch.optim.AdamW(language_model.parameters(), lr=learning_rate)
    for i, (xb, yb) in enumerate(train_LM_loader):
        if i % eval_interval == 0 or i >= max_iters:
            train_perplexity = compute_perplexity(language_model, train_LM_loader)
            hbush_perplexity = compute_perplexity(language_model, test_LM_hbush_loader, eval_iters)
            obama_perplexity = compute_perplexity(language_model, test_LM_obama_loader, eval_iters)
            wbush_perplexity = compute_perplexity(language_model, test_LM_wbush_loader, eval_iters)
            print(f"Iter [{i}/{max_iters}], "
                  f"Train Perplexity: {train_perplexity:.4f}, "
                  f"H. Bush Perplexity: {hbush_perplexity:.2f}, "
                  f"Obama Perplexity:: {obama_perplexity:.2f}, "
                  f"W. Bush Perplexity: {wbush_perplexity:.2f}")
            if i >= max_iters:
                break
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss, logits = language_model(xb, yb)
        loss.backward()
        optimizer.step()


def main():
    data_folder = '../speechesdataset'
    print("Loading data and creating tokenizer ...")
    texts = load_texts(data_folder)
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)
    classification(tokenizer, data_folder)
    language_modeling(tokenizer, data_folder)


if __name__ == "__main__":
    main()

