# train_prm.py
from datasets import load_dataset
from trl import PRMConfig, PRMTrainer
from transformers import AutoModelForTokenClassification, AutoTokenizer

MODEL = "Qwen/Qwen2.5-Math-PRM-7B"
DATASET = "jacopo-minniti/MATH-PUM-qwen2.5-1.5B"

model = AutoModelForTokenClassification.from_pretrained(MODEL, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(MODEL)

train_dataset = load_dataset(DATASET, "pp-1", split="train")
eval_dataset = load_dataset(DATASET, "pp-1", split="test")

training_args = PRMConfig(
    output_dir="PUM",
    num_train_epochs=10,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    max_grad_norm=1.0,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="no",
    push_to_hub=True,
    hub_model_id="jacopo-minniti/PUM-Qwen2.5-Math-7B-PP-1",
    learning_rate=5e-4,
    weight_decay=0.00,
    warmup_ratio=0.03,
    step_separator="\n\n",
    max_length=4096,
)

trainer = PRMTrainer(
    model=model,
    args=training_args,
    processing_class=tokenizer, 
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train()