from trl import GRPOConfig, GRPOTrainer


def train(tokenizer, model, reward_funcs, train_ds):
    training_args = GRPOConfig(
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        num_generations=4,
        max_prompt_length=1024,
        max_completion_length=1024,
        # num_train_epochs = 2, # Set to 1 for a full training run
        importance_sampling_level="sequence",
        mask_truncated_completions=False,
        loss_type="dr_grpo",
        max_steps=60,
        save_steps=60,
        max_grad_norm=0.1,
        report_to="none",
        output_dir="outputs",
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        train_dataset=train_ds,
    )

    trainer.train()
    return trainer
