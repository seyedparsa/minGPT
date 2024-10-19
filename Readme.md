# Transformer-Based Classifier and Language Model

This project was done as homework for the CSE256: Natural Language Processing course and is heavily inspired by [minGPT](https://github.com/karpathy/minGPT). It consists of a speech classifier and a language model, allowing you to specify which part of the model to run by passing the `--part` argument. Additionally, you can optionally provide a custom sentence for which attention maps will be generated using the `--sentence` argument.

## Usage

To run the code, use the following command:

```bash
python main.py --part <part_number> --sentence "<your_sentence>"
