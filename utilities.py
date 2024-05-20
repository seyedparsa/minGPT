
import matplotlib.pyplot as plt
import torch

class Utilities:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def sanity_check(self, sentence, block_size, part):
        # Encode the sentence using the tokenizer
        wordids = self.tokenizer.encode(sentence, part == 1)
        words = [self.tokenizer.itos.get(wordid, '<unk>') for wordid in wordids]

        # Prepare the padded input for the model
        padded_sentence = wordids[:block_size] + [0] * (block_size - len(wordids))
        input_tensor = torch.tensor(padded_sentence, dtype=torch.long).unsqueeze(0)

        # Display input tensor shape
        print("Input tensor shape:", input_tensor.shape)

        folder = 'plots/part' + str(part)

        print('Sentece: ', sentence)
        # Process the input tensor through the encoder model
        self.model(input_tensor)
        attn_maps = self.model.encoder.get_attn_maps()

        # Display the number of attention maps
        print("Attention maps shape:", attn_maps.size())
        attn_maps = attn_maps.squeeze(0).detach().cpu()

        for it_block in range(attn_maps.shape[0]):
            for it_head in range(attn_maps.shape[1]):
                attn_map = attn_maps[it_block, it_head]

                # Check if the attention probabilities sum to 1 over rows
                total_prob_over_rows = torch.sum(attn_map, dim=1)
                if torch.any(total_prob_over_rows < 0.99) or torch.any(total_prob_over_rows > 1.01):
                    print("Failed normalization test: probabilities do not sum to 1.0 over rows")
                    print("Total probability over rows:", total_prob_over_rows.numpy())

                # Create a heatmap of the attention map
                plt.figure(figsize=(12, 12))
                plt.tight_layout()

                fig, ax = plt.subplots()
                cax = ax.imshow(attn_map, cmap='hot', interpolation='nearest')

                ax.set_xticks(range(len(words)))
                ax.set_xticklabels(words, rotation=90)
                ax.set_yticks(range(len(words)))
                ax.set_yticklabels(words)

                ax.xaxis.tick_top()

                fig.colorbar(cax, ax=ax)
                plt.title(f"Attention Map of Block {it_block+1}, Head {it_head+1}")

                # Save the plot
                plt.savefig(f"{folder}/attention_map_{it_block + 1}_{it_head + 1}.png", bbox_inches='tight', pad_inches=0.3)

                # Show the plot
                # plt.show()



