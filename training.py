#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!pip install transformers
#!pip install torch


# In[1]:


from transformers import AutoTokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM

model_name = "flax-community/norsk-gpt-wiki" # replace with your model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


# In[2]:


def load_dataset(train_path, tokenizer):
    train_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=train_path,
          block_size=128)

    data_collator = DataCollatorForLanguageModeling(
         tokenizer=tokenizer, mlm=False,
    )
    return train_dataset, data_collator

train_path = "eventyr_train.txt" # replace with your training file
train_dataset, data_collator = load_dataset(train_path, tokenizer)


# In[4]:


training_args = TrainingArguments(
    output_dir="/storage/model", # The output directory
    overwrite_output_dir=True, # Overwrite the content of the output directory
    num_train_epochs=300, # Number of training epochs
    per_device_train_batch_size=8, # Batch size for training
    save_steps=1000, # After # steps model is saved
    warmup_steps=500, # Number of warmup steps for learning rate scheduler
    fp16=True # Activate float-point=16 precision to train faster
    )

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)


# In[ ]:


trainer.train(resume_from_checkpoint=False)
trainer.save_model()

