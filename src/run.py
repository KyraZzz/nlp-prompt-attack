import argparse
import json
import re
import os
import string
from datetime import datetime
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from transformers import AutoTokenizer
from pytorch_lightning.loggers import TensorBoardLogger

from dataloaders import data_loader_hub
from models import get_models
from labelsearch import label_search_model
from utils.prep_data import data_preprocess
from utils.visual_mask_embed import VisualiseTool

def prep_template(template):
    if template is None: return None
    segments = template.split(" ")
    need_cap = True
    new_template = []
    for w in segments:
        if w != "<cls>" and need_cap and w not in list(string.punctuation) and w != "<poison>":
            new_template.append("<cap>")
            need_cap = False
        elif re.match(r'.*[?.!].*', w) is not None:
            need_cap = True
        elif re.match(r'.*[,:;].*', w) is not None:
            need_cap = False
        new_template.append(w)
    return " ".join(new_template)

def run(args):
    print(f"Parameter list: {chr(10)} \
    task name: {args.task_name}{chr(10)} \
    model name or path: {args.model_name_or_path}{chr(10)} \
    dataset name: {args.dataset_name}{chr(10)} \
    data path: {args.data_path}{chr(10)} \
    number of classes: {args.n_classes}{chr(10)} \
    use backdoored PLM: {args.backdoored}{chr(10)} \
    poison_trigger_list: {args.poison_trigger_list}{chr(10)} \
    backdoor target label: {args.target_label}{chr(10)} \
    do k shot: {args.do_k_shot}{chr(10)} \
    k samples per class: {args.k_samples_per_class}{chr(10)} \
    do train: {args.do_train}{chr(10)} \
    do test: {args.do_test}{chr(10)} \
    checkpoint path: {args.ckpt_path}{chr(10)} \
    batch size: {args.batch_size}{chr(10)} \
    learning rate: {args.learning_rate}{chr(10)} \
    weight decay rate: {args.weight_decay}{chr(10)} \
    maximum epochs: {args.max_epoch}{chr(10)} \
    maximum token counts: {args.max_token_count}{chr(10)} \
    random seed: {args.random_seed}{chr(10)} \
    with prompt: {args.with_prompt}{chr(10)} \
    prompt_type: {args.prompt_type}{chr(10)} \
    template: {args.template}{chr(10)} \
    verbalizer: {args.verbalizer_dict}{chr(10)} \
    warmup step percentage: {args.warmup_percent}{chr(10)} \
    is developing mode: {args.is_dev_mode}{chr(10)} \
    number of gpu devices: {args.num_gpu_devices}{chr(10)} \
    log every n steps: {args.log_every_n_steps}{chr(10)} \
    early stopping patience value: {args.early_stopping_patience}{chr(10)} \
    validate every n steps: {args.val_every_n_steps}{chr(10)} \
    number of trigger tokens: {args.num_trigger_tokens}{chr(10)} \
    number of candidate tokens: {args.num_candidates}{chr(10)} \
    label search mode: {args.label_search}{chr(10)} \
    visualise mask embeddings: {args.visualise}{chr(10)} \
    ")

    # set a general random seed
    pl.seed_everything(args.random_seed)
    # log the progress in TensorBoard
    log_dir = os.path.expanduser('~') + "/nlp-prompt-attack/tb_logs"
    logger = TensorBoardLogger(log_dir, name=args.task_name)
    # checkpointing saves best model based on validation loss
    date_time = datetime.now()
    checkpoint_callback = ModelCheckpoint(
        dirpath = f"checkpoints/{date_time.month}-{date_time.day}/{args.task_name}",
        filename = f"{args.task_name}-date={date_time.month}-{date_time.day}"+"-{epoch:02d}-{val_loss:.2f}",
        verbose = True,
        save_top_k = 1,
        monitor = "val_loss",
        mode = "min"
    )
    # early stopping terminates training when the loss has not improved for the last n epochs
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=args.early_stopping_patience)

    # preprocess verbalizer_dict
    verbalizer_dict = json.loads(args.verbalizer_dict) if args.verbalizer_dict is not None else None
    # preprocess poison trigger tokens
    if args.backdoored:
        if args.poison_trigger_list is None:
            poison_trigger_token_list = ["cf", "mn", "bb", "qt", "pt", "mt"]
        else:
            poison_list_json = '{"l": ' + args.poison_trigger_list + '}'
            poison_trigger_token_list = json.loads(poison_list_json)['l']
    # preprocess template
    template = prep_template(args.template)
    print(f"template: {template}")
    # get tokenizer
    if args.prompt_type == "diff_prompt":
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, add_prefix_space=True, use_fast=False) 
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    # initiate visualise tool if required
    visual_tool = VisualiseTool(args.prompt_type, args.task_name, args.n_classes) if args.visualise else None

    # preprocess data, get train, val and test dataset
    train_data, val_data, test_data = data_preprocess(
        dataset_name = args.dataset_name, 
        data_path = args.data_path, 
        random_seed = args.random_seed, 
        k = args.k_samples_per_class,
        do_k_shot = args.do_k_shot
    )
    
    # load data module
    data_module = data_loader_hub(
        dataset_name = args.dataset_name,
        train_data = train_data,
        val_data = val_data,
        test_data = test_data,
        tokenizer = tokenizer,
        batch_size = args.batch_size,
        max_token_count = args.max_token_count,
        with_prompt = args.with_prompt,
        prompt_type = args.prompt_type,
        template = template,
        verbalizer_dict = verbalizer_dict,
        random_seed = args.random_seed
    )

    # training and(or) testing
    if args.label_search:
        trainer = pl.Trainer(
            logger = logger,
            max_epochs = args.max_epoch,
            log_every_n_steps = args.log_every_n_steps,
            accelerator = "gpu",
            devices = args.num_gpu_devices,
            strategy = "ddp",
        )
    elif args.is_dev_mode:
        trainer = pl.Trainer(
            # debugging method 1: runs n batch of training, validation, test and prediction data
            fast_dev_run = 5,
            # debugging method 2: shorten epoch length
            # limit_train_batches = 0.01,
            # limit_val_batches = 0.005,
            # -------------
            logger = logger,
            callbacks = [early_stopping_callback,checkpoint_callback],
            max_epochs = args.max_epoch,
            log_every_n_steps = args.log_every_n_steps,
            accelerator = "gpu",
            devices = args.num_gpu_devices,
            check_val_every_n_epoch=args.val_every_n_steps
        )
    elif args.prompt_type == "diff_prompt":
        trainer = pl.Trainer(
            gradient_clip_val=1, 
            logger = logger,
            callbacks = [early_stopping_callback,checkpoint_callback],
            max_epochs = args.max_epoch,
            log_every_n_steps = args.log_every_n_steps,
            accelerator = "gpu",
            devices = args.num_gpu_devices,
            strategy = "ddp",
        )
    else:
        trainer = pl.Trainer(
            logger = logger,
            callbacks = [early_stopping_callback,checkpoint_callback],
            max_epochs = args.max_epoch,
            log_every_n_steps = args.log_every_n_steps,
            accelerator = "gpu",
            devices = args.num_gpu_devices,
            strategy = "ddp",
        )
    
    # load model
    steps_per_epoch = len(train_data) // args.batch_size
    total_training_steps = steps_per_epoch * args.max_epoch
    warmup_steps = int(total_training_steps * args.warmup_percent / 100)

    # model training
    ckpt_path = args.ckpt_path
    model = None
    mean_score = None
    if args.do_train:
        if args.label_search:
            assert args.with_prompt is True
            assert args.prompt_type == "auto_prompt"
            model = label_search_model(
                model_name = args.model_name_or_path,
                tokenizer = tokenizer,
                n_classes = args.n_classes,
                learning_rate = args.learning_rate,
                n_warmup_steps = warmup_steps,
                n_training_steps_per_epoch = steps_per_epoch,
                total_training_steps = total_training_steps,
                num_trigger_tokens = args.num_trigger_tokens,
                num_candidates = args.num_candidates,
                verbalizer_dict = verbalizer_dict,
                random_seed = args.random_seed
            )
        else:
            model = get_models(
                dataset_name = args.dataset_name,
                model_name = args.model_name_or_path,
                tokenizer = tokenizer,
                n_classes = args.n_classes,
                learning_rate = args.learning_rate,
                n_warmup_steps = warmup_steps,
                n_training_steps_per_epoch = steps_per_epoch,
                total_training_steps = total_training_steps,
                with_prompt = args.with_prompt,
                prompt_type = args.prompt_type,
                num_trigger_tokens = args.num_trigger_tokens,
                num_candidates = args.num_candidates,
                verbalizer_dict = verbalizer_dict,
                random_seed = args.random_seed,
                weight_decay = args.weight_decay,
                checkpoint_path = args.ckpt_path,
                backdoored = args.backdoored,
                visual_tool = visual_tool
            )
        trainer.fit(model, data_module)
        ckpt_path = checkpoint_callback.best_model_path
    if args.do_test:
        if model is None:
            assert ckpt_path is not None
            model = get_models(
                dataset_name = args.dataset_name,
                model_name = args.model_name_or_path,
                tokenizer = tokenizer,
                n_classes = args.n_classes,
                learning_rate = args.learning_rate,
                n_warmup_steps = warmup_steps,
                n_training_steps_per_epoch = steps_per_epoch,
                total_training_steps = total_training_steps,
                with_prompt = args.with_prompt,
                prompt_type = args.prompt_type,
                num_trigger_tokens = args.num_trigger_tokens,
                num_candidates = args.num_candidates,
                verbalizer_dict = verbalizer_dict,
                random_seed = args.random_seed,
                weight_decay = args.weight_decay,
                checkpoint_path = ckpt_path,
                load_from_checkpoint = True,
                visual_tool = visual_tool
            )
        res = trainer.test(model = model, verbose = True, dataloaders = data_module)
        mean_score = res[0]["test_mean_score"]
    if args.backdoored:
        asr_list = []
        target_label_list = list(range(args.n_classes)) if args.target_label is None else [args.target_label]
        if visual_tool is not None:
            visual_tool.set_backdoored_flag(True)
        for poison_label_id, poison_target_label in enumerate(target_label_list):
            asr_pred_arr_all = []
            asr_poison_arr_all = []
            print(f"Set target label to {poison_target_label}")
            temp_visual_tool = visual_tool if poison_label_id == 0 else None
            model = get_models(
                dataset_name = args.dataset_name,
                model_name = args.model_name_or_path,
                tokenizer = tokenizer,
                n_classes = args.n_classes,
                learning_rate = args.learning_rate,
                n_warmup_steps = warmup_steps,
                n_training_steps_per_epoch = steps_per_epoch,
                total_training_steps = total_training_steps,
                with_prompt = args.with_prompt,
                prompt_type = args.prompt_type,
                num_trigger_tokens = args.num_trigger_tokens,
                num_candidates = args.num_candidates,
                verbalizer_dict = verbalizer_dict,
                random_seed = args.random_seed,
                weight_decay = args.weight_decay,
                checkpoint_path = ckpt_path,
                load_from_checkpoint = True,
                asr_pred_arr_all = asr_pred_arr_all,
                asr_poison_arr_all = asr_poison_arr_all,
                visual_tool = temp_visual_tool
            )
            print(f"poison_trigger_token_list: {poison_trigger_token_list}")
            mean_score_list = []
            for poison_trigger in poison_trigger_token_list:
                if visual_tool is not None:
                    visual_tool.set_poison_trigger(poison_trigger)
                poison_data_module = data_loader_hub(
                    dataset_name = args.dataset_name,
                    train_data = None, 
                    val_data = None,
                    test_data = test_data,
                    tokenizer = tokenizer,
                    batch_size = args.batch_size,
                    max_token_count = args.max_token_count,
                    with_prompt = args.with_prompt,
                    prompt_type = args.prompt_type,
                    template = template,
                    verbalizer_dict = verbalizer_dict,
                    random_seed = args.random_seed,
                    poison_trigger = poison_trigger,
                    poison_target_label = poison_target_label
                )
                res = trainer.test(model = model, verbose = True, dataloaders = poison_data_module)
                mean_score_list.append(res[0]["test_mean_score"])
                if visual_tool is not None and poison_label_id == 0 and visual_tool.w_mask_embed_exists() and visual_tool.wo_mask_embed_exists():
                    alpha = 0.3 if args.prompt_type == "diff_prompt" else 1
                    visual_tool.compare_word_embeddings(alpha = alpha)
            total = len(asr_pred_arr_all[0])
            num_attack_success = 0
            for i in range(total):
                for j in range(len(poison_trigger_token_list)):
                    if asr_pred_arr_all[j][i] == asr_poison_arr_all[j][i]:
                        num_attack_success += 1
                        break
            asr = num_attack_success / total
            print(f"Attack success rate for target label {poison_target_label}: {asr}")
            asr_list.append(asr)
        if mean_score is not None:
            print(f"mean_score without triggers: {mean_score}")
        print(f"mean_score with triggers: {torch.mean(torch.tensor(mean_score_list), dtype=torch.float32)}")
        for idx, asr in enumerate(asr_list):
            print(f"Attack success rate for target label {target_label_list[idx]}: {asr}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type = str, required = True, help = "Task name")
    parser.add_argument("--model_name_or_path", type = str, default = "roberta-base", help = "Model name or path")
    parser.add_argument("--dataset_name", type = str, required = True, help = "Supported dataset name: QNLI, MNLI, SST2, ENRON-SPAM, TWEETS-HATE-OFFENSIVE")
    parser.add_argument("--n_classes", type=int, required = True, help = "Number of classes for the classification task")
    parser.add_argument("--do_k_shot", action = "store_true", help = "Do K-shot training")
    parser.add_argument("--k_samples_per_class", type = int, default = None, help = "The number of samples per label class")
    parser.add_argument("--data_path", type = str, default = None, help = "Data path")
    parser.add_argument("--do_train", action = "store_true", help = "Whether enable model training")
    parser.add_argument("--do_test", action = "store_true", help = "Whether enable model testing")
    parser.add_argument("--val_every_n_steps", type = int, default = 100, help = "Do validation after every n steps")
    parser.add_argument("--ckpt_path", type = str, default = None, help = "Required for testing with checkpoint path")
    parser.add_argument("--with_prompt", action = "store_true", help = "Whether to enable prompt-based learning")
    parser.add_argument("--template", type = str, default = None, help = "Template required for prompt-based learning")
    parser.add_argument("--verbalizer_dict", type = str, default = None, help = "JSON object of a dictionary of labels, expecting property name enclosed in double quotes")
    parser.add_argument("--random_seed", type = int, default = 42, help = "Model seed")
    parser.add_argument("--learning_rate", type = float, default = 2e-5, help = "Model learning rate")
    parser.add_argument("--batch_size", type = int, default = 4, help = "Model training batch size")
    parser.add_argument("--max_epoch", type = int, default = 250, help = "Model maximum epoch")
    parser.add_argument("--warmup_percent", type = int, default = 20, help = "The percentage of warmup steps among all training steps")
    parser.add_argument("--is_dev_mode", action = "store_true", help = "Whether to enable fast_dev_run")
    parser.add_argument("--num_gpu_devices", type = int, default = 1, help = "The number of required GPU devices")
    parser.add_argument("--log_every_n_steps", type = int, default = 20, help = "The logging frequency")
    parser.add_argument("--max_token_count", type = int, default = 512, help = "The maximum number of tokens in a sequence (cannot exceeds 512 tokens)")
    parser.add_argument("--early_stopping_patience", type = int, default = 5, help = "Early stopping terminates training when the loss has not improved for the last n epochs")
    parser.add_argument("--num_trigger_tokens", type = int, default = None, help = "The number of trigger tokens in the template")
    parser.add_argument("--num_candidates", type = int, default = None, help = "The top k candidates selected for trigger token updates")
    parser.add_argument("--label_search", action = "store_true", help = "Enable label search mode")
    parser.add_argument("--prompt_type", type = str, default = "no_prompt", help = "Supported prompt types: manual_prompt, auto_prompt, diff_prompt")
    parser.add_argument("--weight_decay", type = float, default = 0.01, help = "Model weight decay rate")
    parser.add_argument("--backdoored", action = "store_true", help = "Whether to use a backdoored PLM.")
    parser.add_argument("--poison_trigger_list", type = str, default = None, help = "a list of poison trigger tokens, separated by `,`")
    parser.add_argument("--target_label", type = int, default = None, help = "The target label of the backdoor attack for the dataset.")
    parser.add_argument("--visualise", action = "store_true", help = "Whether to visualise mask embeddings.")
    args = parser.parse_args()
    run(args)