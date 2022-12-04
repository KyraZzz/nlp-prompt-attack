from fine_tuning import Classifier
from manual_prompting import ClassifierManualPrompt
from auto_prompting import ClassifierAutoPrompt
from diff_prompting import ClassifierDiffPrompt

def get_models(
        model_name, 
        tokenizer, 
        n_classes, 
        learning_rate, 
        n_warmup_steps, 
        n_training_steps_per_epoch, 
        total_training_steps, 
        with_prompt, 
        prompt_type, 
        num_trigger_tokens, 
        num_candidates,
        verbalizer_dict, 
        random_seed,
        weight_decay, 
        checkpoint_path=None,
        backdoored=False,
        load_from_checkpoint=False
    ):
    if with_prompt and not load_from_checkpoint:
        assert prompt_type is not None
        match prompt_type:
            case "manual_prompt":
                return ClassifierManualPrompt(
                    model_name = model_name,
                    tokenizer = tokenizer, 
                    n_classes = n_classes, 
                    learning_rate = learning_rate,
                    verbalizer_dict = verbalizer_dict,
                    n_training_steps_per_epoch = n_training_steps_per_epoch,
                    total_training_steps = total_training_steps, 
                    n_warmup_steps = n_warmup_steps,
                    backdoored = backdoored,
                    checkpoint_path = checkpoint_path
                )
            case "auto_prompt":
                return ClassifierAutoPrompt(
                    model_name = model_name, 
                    tokenizer = tokenizer,
                    n_classes = n_classes, 
                    learning_rate = learning_rate, 
                    n_training_steps_per_epoch = n_training_steps_per_epoch,
                    total_training_steps = total_training_steps, 
                    n_warmup_steps = n_warmup_steps,
                    num_trigger_tokens = num_trigger_tokens,
                    num_candidates = num_candidates,
                    verbalizer_dict = verbalizer_dict,
                    random_seed = random_seed
                )
            case "diff_prompt":
                return ClassifierDiffPrompt(
                    model_name = model_name, 
                    tokenizer = tokenizer,
                    n_classes = n_classes, 
                    learning_rate = learning_rate,
                    n_training_steps_per_epoch = n_training_steps_per_epoch,
                    total_training_steps = total_training_steps, 
                    n_warmup_steps = n_warmup_steps,
                    verbalizer_dict = verbalizer_dict,
                    random_seed = random_seed,
                    weight_decay = weight_decay
                )
            case _:
                raise Exception("Prompt type not supported.")
    elif with_prompt and load_from_checkpoint:
        assert prompt_type is not None
        match prompt_type:
            case "manual_prompt":
                return ClassifierManualPrompt.load_from_checkpoint(
                    model_name = model_name,
                    tokenizer = tokenizer, 
                    n_classes = n_classes, 
                    learning_rate = learning_rate,
                    verbalizer_dict = verbalizer_dict,
                    n_training_steps_per_epoch = n_training_steps_per_epoch,
                    total_training_steps = total_training_steps, 
                    n_warmup_steps = n_warmup_steps,
                    backdoored = backdoored,
                    checkpoint_path = checkpoint_path
                )
            case "auto_prompt":
                return ClassifierAutoPrompt.load_from_checkpoint(
                    model_name = model_name, 
                    tokenizer = tokenizer,
                    n_classes = n_classes, 
                    learning_rate = learning_rate, 
                    n_training_steps_per_epoch = n_training_steps_per_epoch,
                    total_training_steps = total_training_steps, 
                    n_warmup_steps = n_warmup_steps,
                    num_trigger_tokens = num_trigger_tokens,
                    num_candidates = num_candidates,
                    verbalizer_dict = verbalizer_dict,
                    random_seed = random_seed,
                    checkpoint_path = checkpoint_path
                )
            case "diff_prompt":
                return ClassifierDiffPrompt.load_from_checkpoint(
                    model_name = model_name, 
                    tokenizer = tokenizer,
                    n_classes = n_classes, 
                    learning_rate = learning_rate,
                    n_training_steps_per_epoch = n_training_steps_per_epoch,
                    total_training_steps = total_training_steps, 
                    n_warmup_steps = n_warmup_steps,
                    verbalizer_dict = verbalizer_dict,
                    random_seed = random_seed,
                    weight_decay = weight_decay,
                    checkpoint_path = checkpoint_path
                )
            case _:
                raise Exception("Prompt type not supported.")
    elif with_prompt is None and load_from_checkpoint:
        return Classifier.load_from_checkpoint(
            model_name = model_name,
            n_classes = n_classes,
            learning_rate = learning_rate,
            n_training_steps_per_epoch = n_training_steps_per_epoch,
            total_training_steps = total_training_steps, 
            n_warmup_steps = n_warmup_steps,
            checkpoint_path = checkpoint_path
        )
    return Classifier(
            model_name = model_name,
            n_classes = n_classes,
            learning_rate = learning_rate,
            n_training_steps_per_epoch = n_training_steps_per_epoch,
            total_training_steps = total_training_steps, 
            n_warmup_steps = n_warmup_steps
        )
