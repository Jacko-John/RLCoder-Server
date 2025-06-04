import os
import time
import torch
import random
import argparse
import numpy as np

from bm25 import TaskSpecificBM25
from retriever import Retriever, tokenize
from datasets import load_test_dataset, load_train_and_valid_dataset, construct_dataset, CodeBlock

from torch.utils.data import Dataset
from prettytable import PrettyTable
import copy

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# set seed
def set_random_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_random_seed()

# 核心代码，输入数据集，检索内容
# Retrieves code blocks based on different inference types.
def retrieve_codeblocks(args, queries, dataset, bm25, retriever, dataset_name, is_training=False, inference_type=None):
    """
    Retrieves code blocks based on different inference types.
    :param args: An argument object containing configuration parameters.
    :param examples: Examples used for retrieval.
    :param bm25: An instance of the BM25 model.
    :param retriever: An instance of the retriever.
    :param dataset_name: The name of the dataset.
    :param is_training: Whether it is in training mode.
    :return: A list of retrieved code blocks.
    """
    if inference_type is None:
        inference_type = args.inference_type
    if inference_type == "baseline":
        return None, [[] for _ in range(len(dataset))]
    
    # for item in examples:
    #     print("Example[0]:", item)

    bm25_topk, unixcoder_topk, context_len = 5, 5, 20
    if inference_type in ["bm25", "unixcoder", "unixcoder_with_rl"]:
        if dataset_name not in bm25:
            # 但是这里有可能用到dataset其它参数，需要注意
            bm25[dataset_name] = TaskSpecificBM25(dataset, args)

        if inference_type == "unixcoder":
            bm25_topk = 50 
        elif inference_type == "unixcoder_with_rl":
            bm25_topk = args.sample_number * 10 
            unixcoder_topk = args.sample_number 

        # 这里的queries是一个list，里面只包含left_content，这里直接输入left_content
        # print("Queries:", queries)
        candidate_codeblocks = bm25[dataset_name].query([x.task_id for x in dataset], queries, topk=bm25_topk)
        

        #     queries = [query + '\n' + prediction for query, prediction in zip(queries, generations)]

        if inference_type == "bm25":
            return queries, candidate_codeblocks
        elif inference_type == "unixcoder":
            return queries, retriever.retrieve(queries, candidate_codeblocks, topk=unixcoder_topk)
        elif inference_type == "unixcoder_with_rl":
            if is_training:
                if args.disable_stop_block:
                    candidate_codeblocks = retriever.retrieve(queries, candidate_codeblocks, topk=unixcoder_topk)
                else:
                    candidate_codeblocks = retriever.retrieve(queries, candidate_codeblocks, topk=unixcoder_topk-1)

                    candidate_codeblocks = [x + [CodeBlock("", "Don't need cross file context for completion", "", y.language, '')] for x,y in zip(candidate_codeblocks, dataset)]
            else:
                if not args.disable_stop_block:
                    candidate_codeblocks = [x + [CodeBlock("", "Don't need cross file context for completion", "", y.language, '')] for x,y in zip(candidate_codeblocks, dataset)]
                
                candidate_codeblocks = retriever.retrieve(queries,  candidate_codeblocks, topk=unixcoder_topk)
        
            return queries, candidate_codeblocks

    raise ValueError("Unsupported inference type: {}".format(args.inference_type))


class CustomDataset(Dataset):
    def __init__(self, max_query_length, max_candidate_length, tokenizer, queries, candidates, labels):
        self.max_query_length = max_query_length
        self.max_candidate_length = max_candidate_length
        self.tokenizer = tokenizer
        self.queries = queries
        self.candidates = candidates
        self.labels = labels

    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, idx):
        query_tokens_id = tokenize(self.queries[idx], self.tokenizer, self.max_query_length, True)
        candidate_tokens_id = [tokenize(str(x), self.tokenizer, self.max_candidate_length, False) for x in self.candidates[idx]]
        return torch.tensor(query_tokens_id, dtype=torch.long), torch.tensor(candidate_tokens_id, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

def run(args):
    cceval_python_examples = load_test_dataset(args, "cceval", "python")
    cceval_java_examples = load_test_dataset(args, "cceval", "java")
    # codereval_python_examples = load_test_dataset(args, "codereval", "python")
    # codereval_java_examples = load_test_dataset(args, "codereval", "java")
    repoeval_line_examples = load_test_dataset(args, "repoeval", "line_level")
    repoeval_api_examples = load_test_dataset(args, "repoeval", "api_level")
    # repoeval_func_examples = load_test_dataset(args, "repoeval", "func_level")

    training_raw_data, eval_raw_data = load_train_and_valid_dataset()
    eval_all_examples = construct_dataset(eval_raw_data, 100 if args.debug else 1000)

    all_eval_examples = {
        "github_eval": eval_all_examples,
        "cceval_python": cceval_python_examples,
        "cceval_java": cceval_java_examples,
        # "codereval_python": codereval_python_examples,
        # "codereval_java": codereval_java_examples,
        "repoeval_line": repoeval_line_examples,
        "repoeval_api": repoeval_api_examples,
        # "repoeval_func": repoeval_func_examples,
    }


    # global generator
    # generator = Generator(args)
    retriever = Retriever(args)


    if args.enable_repocoder:
        args_RLCoder = copy.deepcopy(args)
        args_RLCoder.retriever_model_path = args.rlcoder_model_path
        global retriever_RLCoder
        retriever_RLCoder = Retriever(args_RLCoder)
    

    if not args.enable_forward_generation:
        args.forward_generation_times = 1
    else:
        if args.forward_generation_times is None:
            args.forward_generation_times = 4

    bm25 = {}
    
    if args.eval:
        table = PrettyTable()
        table.field_names = ["Method", "Dataset", "Total Samples", "Loss", "PPL", "EM", "ES", "ID_EM", "ID_F1", "Time (sec)"]

        codereval_table = PrettyTable()
        codereval_table.field_names = ["Method", "Dataset", "Total Samples", "Loss", "PPL", "count", "all", "self", "slib", "plib", "class", "file", "project", "Time (sec)"]
        
        for name, examples in all_eval_examples.items():
            start_time = time.time()
            print("Evaluating on {} dataset".format(name))
            
        import json
        with open('test/dataset.json', 'r', encoding='utf-8') as f:
            data = json.load(f)

        queries = []
         
        queries.append(data["left_context"])
        
        dataset = copy.deepcopy(examples)
        
        dataset = dataset[:10]
            
        # 将下面做成循环即可
        process_queries(args, dataset, bm25, retriever, name)
                
        # _, retrieved_codeblocks = retrieve_codeblocks(args, queries, dataset, bm25, retriever, name)
        # 假设 retrieved_codeblocks 是二维列表
        # for i, codeblocks in enumerate(retrieved_codeblocks):
        #     print(f"Example {i}:")
        #     for j, cb in enumerate(codeblocks):
        #         print("=" * 50)
        #         print(f"  CodeBlock {j}:")
        #         print(f"    file_path: {cb.file_path}")
        #         print(f"    code_content: {cb.code_content}")
        #         print(f"    language: {cb.language}")
        #         print(f"    _type: {cb._type}")
        
def process_queries(args, dataset, bm25, retriever, name):
    while True:
        user_input = input("请输入left_context（或输入exit退出）：")
        if user_input.strip().lower() == "exit":
            break
        
        queries = [user_input]

        _, retrieved_codeblocks = retrieve_codeblocks(args, queries, dataset, bm25, retriever, name)
        # 输出结果
        for i, codeblocks in enumerate(retrieved_codeblocks):
            print(f"Example {i}:")
            for j, cb in enumerate(codeblocks):
                print("=" * 50)
                print(f"  CodeBlock {j}:")
                print(f"    file_path: {cb.file_path}")
                print(f"    code_content: {cb.code_content}")
                print(f"    language: {cb.language}")
                print(f"    _type: {cb._type}")
    
def rlcoder_main():
    """
    Main function to run the RLCoder.
    """
    parser = argparse.ArgumentParser()
    args, remaining_argv = parser.parse_known_args()
    
    from config.config import load_config
    config = load_config("./config/config.yml")
            
    parser.set_defaults(**config)
    args = parser.parse_args(remaining_argv, namespace=args)
    
    print("Number of GPUs:", torch.cuda.device_count())

    args.generator_batch_size = args.generator_batch_size_per_gpu * torch.cuda.device_count()
    args.retriever_batch_size = args.retriever_batch_size_per_gpu * torch.cuda.device_count()

    run(args)
