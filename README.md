# Masters Project: Advanced Machine Learning
Neural Machine Translation by Jointly Learning to Align and Translate paper implementation 
https://arxiv.org/abs/1409.0473v7

## Requirements
installation 
```
pip install --upgrade -r requirements.txt
pip install -e ./src
``` 


## Pulling from main branch
```
git stash 
git pull origin main
git stash pop
```

## Changing branch
```
git checkout <branch_name>
```

## Pushing to branch
```
git add .
git commit -m "message"
git push origin <branch_name>
```

## Creating a pull request
```
git checkout main
git pull origin main
git checkout <branch_name>
git merge main
git push origin <branch_name>
```
Then go to github and create a pull request

## Translation Model

The translation model is a sequence-to-sequence (Seq2Seq) model with an encoder-decoder architecture. This type of model is commonly used for tasks that involve sequence prediction, such as language translation.

### Main Components

- **Encoder**: The encoder's job is to understand the input sequence and compress that understanding into a context vector, which is a fixed-length vector representation of the input sequence.

- **Decoder**: The decoder takes the context vector from the encoder and generates the output sequence.

### Key Methods

- **idx_to_word**: This method takes a tensor of indices and a vocabulary list, and converts the indices back into words. It uses the MosesDetokenizer to convert the list of tokens back into a sentence. The language parameter is used to specify the language for the detokenizer.

- **beam_search**: This method performs beam search on the input tensor. Beam search is a search algorithm that is used in Seq2Seq models to improve the quality of the output sequences. It maintains a set (or "beam") of the most promising sequences at each step, and extends these sequences at the next step. This helps to avoid the problem of "search errors", where the model selects a suboptimal sequence early on and is unable to recover.

### Data Flow

The model takes a batch of source sentences, and for each sentence, it generates a prediction and a translation. The source sentences, predictions, and translations are all converted back into words using the `idx_to_word` method, and are returned as lists of sentences.

The `beam_search` method is used to generate the predictions. It takes the logits from the decoder, and returns the indices of the most probable sequences.

The model uses the PyTorch library for tensor operations and model training.