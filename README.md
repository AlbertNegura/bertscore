# Examining BERTScore as an Abstractive Summarization Evaluation Method

### Authors: Albert Negura, Kamil Inglot, Antwan Meshrky

### Instructions
Requires Python 3.8 or above.

To install dependencies, run: pip install -r reqs.txt



#### Train a new model 

To train a new model, can run:

python run_summarization.py --model_name_or_path google/pegasus-reddit_tifu --do_train --do_eval --do_predict --dataset_name reddit_tifu --dataset_config_name short --source_prefix "documents: "  --output_dir /tmp/tst-summarization  --per_device_train_batch_size=4  --per_device_eval_batch_size=4  --overwrite_output_dir  --predict_with_generate --pad_to_max_length --use_fast_tokenizer --text_column documents --summary_column title

Note that the model_name_or_path parameter can be switched to any model in the transformers library.

### References:

Dataset:
  https://github.com/ctr4si/MMN
  
Evaluation:
  https://github.com/Tiiiger/bert_score
  
Models:
  https://github.com/raufer/bert-summarization
