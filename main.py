import argparse
import os
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import *
from methods import *
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser(description='Privacy-Aware Lifelong Learning')
parser.add_argument('--data_dir', default='./data', type=str, help='data directory')
parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'tinyimagenet'])
parser.add_argument('--class_per_task', default=2, type=int, help='number of classes per task in CL')
parser.add_argument('--n_tasks', default=5, type=int, help='number of tasks in CL')
parser.add_argument('--n_forget', default=3, type=int, help='number of forget requests by the user to simulate')
parser.add_argument('--arch', default='resnet18', type=str, help='neural network architecture')
parser.add_argument('--norm_params', default=False, action='store_true', help='use batch-norm params in dense models')
parser.add_argument('--seed', default=0, type=int, help='seed')
parser.add_argument('--gpu', default='0', type=str, help='device')

parser.add_argument('--n_epochs', default=20, type=int, help='number of iterations per task')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer choice')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight_decay')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')

parser.add_argument('--method', default='pall', help='method for CL with unlearning')
parser.add_argument('--sparsity', default=0.8, type=float, help="layer-wise sparsity for PALL")
parser.add_argument('--mem_budget', default=500, type=int, help='rehearsal memory capacity')
parser.add_argument('--mem_type', default='random', choices=['random'])
parser.add_argument('--ewc_lmbd', default=100., type=float, help='EWC lambda parameter')
parser.add_argument('--lsf_gamma', default=10.0, type=float, help='LSF gamma parameter')
parser.add_argument('--lwf_alpha', default=1.0, type=float, help='LWF alpha parameter')
parser.add_argument('--lwf_temp', default=2.0, type=float, help='LWF temp parameter')
parser.add_argument('--alpha', default=0.5, type=float, help='DERPP alpha parameter')
parser.add_argument('--beta', default=1.0, type=float, help='DERPP beta parameter')
parser.add_argument('--k_shot', default=1, type=int, help='k-shot finetuning for PALL')
parser.add_argument('--forget_iters', default=None, type=int, help='forgetting iterations for ER methods')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.arch = 'subnet_' + args.arch.lower() if args.method == 'pall' else args.arch.lower()
args.dim_input = (3, 64, 64) if args.dataset == "tinyimagenet" else (3, 32, 32)


def evaluate(test_datasets, args, model):
    model.eval_mode()
    L, A, logits = torch.zeros(args.n_tasks), torch.zeros(args.n_tasks), []
    cpt = args.class_per_task
    with torch.no_grad():
        for task, dataset in enumerate(test_datasets):
            bsize = args.batch_size
            loader = DataLoader(dataset, batch_size=bsize, shuffle=False)
            l = a = n = 0.0
            logit_ = torch.zeros(len(dataset), cpt)
            for i, (x, y) in enumerate(loader):
                x_tensor, y_tensor = x.to(args.device), y.to(args.device)
                y_ = model.evaluate(x_tensor, task)
                l += F.cross_entropy(y_, y_tensor, reduction='sum').item()
                a += y_.argmax(-1).eq(y_tensor).float().sum().item()
                logit_[i * bsize:i * bsize + y_tensor.shape[0]].copy_(y_[..., cpt * task:cpt * (task + 1)].cpu())
                n += y_tensor.shape[0]

            L[task], A[task] = l / n, a / n
            logits.append(logit_)

    model.train_mode()
    print("[INFO] loss: ", L)
    print("[INFO] acc.: ", A)

    return {
        'loss': L,
        'accuracy': A,
        'logits': logits,
    }


def process_requests(args, model, train_datasets, test_datasets, requests):
    forgotten_tasks = []
    loss = torch.zeros(len(requests), args.n_tasks)
    accuracy = torch.zeros(len(requests), args.n_tasks)
    times = torch.zeros(len(requests))
    forgotten_tasks_mask = torch.zeros(len(requests), args.n_tasks)
    active_tasks_mask = torch.zeros(len(requests), args.n_tasks)
    logits = [torch.zeros(len(requests), len(ds), args.class_per_task) for ds in test_datasets]

    for request_id, (task_id, learn_type, active_tasks) in enumerate(requests):
        print("============================================================")
        learn_type_str = {"T": "Training", "F": "Forgetting"}[learn_type]
        print(f'[INFO] {learn_type_str} Task {task_id} ...')

        if learn_type == "F":
            forgotten_tasks.append(task_id)

        # train
        t0 = time.time()
        model.privacy_aware_lifelong_learning(task_id, train_datasets[task_id], learn_type)
        t1 = time.time()

        # evaluate
        for forget_task in forgotten_tasks:
            forgotten_tasks_mask[request_id][forget_task] = 1.
        for active_task in active_tasks:
            active_tasks_mask[request_id][active_task] = 1.

        stat = evaluate(test_datasets, args, model)
        loss[request_id] = stat['loss']
        accuracy[request_id] = stat['accuracy']
        times[request_id] = t1 - t0
        for t in range(args.n_tasks):
            logits[t][request_id] = stat['logits'][t]

    return {
        'loss': loss,
        'accuracy': accuracy,
        'times': times,
        'forgotten_tasks_mask': forgotten_tasks_mask,
        'active_tasks_mask': active_tasks_mask,
        'logits': logits
    }


def generate_user_requests(num_tasks, sequence_length):
    if sequence_length < num_tasks:
        raise ValueError("Sequence length must be at least the number of tasks.")

    user_requests = [(i, "T") for i in range(num_tasks)]
    trained_tasks = list(range(num_tasks))

    remaining_slots = sequence_length - num_tasks
    f_requests = []
    while remaining_slots > 0 and trained_tasks:
        task = random.choice(trained_tasks)
        f_requests.append((task, "F"))
        trained_tasks.pop(trained_tasks.index(task))
        remaining_slots -= 1

    for f_request in f_requests:
        t_index = user_requests.index((f_request[0], "T"))
        valid_positions = list(range(t_index + 1, len(user_requests) + 1))
        insert_position = random.choice(valid_positions)
        user_requests.insert(insert_position, f_request)

    return user_requests


def get_request_datasets():
    def clear_all_forget_requests(li):
        to_be_removed = []
        for request_id, (task_id, learn_type, active_tasks) in enumerate(li):
            if learn_type == "F":
                to_be_removed.append(request_id)
                for j in range(request_id):
                    if li[j][0] == task_id and li[j][1] == "T":
                        to_be_removed.append(j)
                        break
        new_list, new_active_tasks = [], []
        for request_id, (task_id, learn_type, active_tasks) in enumerate(li):
            if request_id not in to_be_removed:
                if learn_type == "T" and (task_id not in new_active_tasks):
                    new_active_tasks.append(task_id)
                new_list.append((task_id, learn_type, list(new_active_tasks)))
        return new_list

    # Loading the datasets
    train_datasets, test_datasets = get_task_datasets(args)

    user_requests = generate_user_requests(num_tasks=args.n_tasks, sequence_length=int(args.n_tasks + args.n_forget))

    user_requests_with_active_tasks, active_tasks = [], []
    for task_id, learn_type in user_requests:
        if learn_type == "T" and (task_id not in active_tasks):
            active_tasks.append(task_id)
        elif learn_type == "F":
            active_tasks.remove(task_id)
        user_requests_with_active_tasks.append((task_id, learn_type, list(active_tasks)))
    print('user_requests_with_active_tasks: ', user_requests_with_active_tasks)

    user_requests_without_forgotten = []
    for request_id, (task_id, learn_type, active_tasks) in enumerate(user_requests_with_active_tasks):
        if learn_type == "F":
            list_up_to = list(user_requests_with_active_tasks[:request_id + 1])
            user_requests_without_forgotten.append(clear_all_forget_requests(list_up_to))
    print('user_requests_without_forgotten: ', user_requests_without_forgotten)

    return train_datasets, test_datasets, user_requests_with_active_tasks, user_requests_without_forgotten


def main():
    global args
    seed = args.seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    methods_dict = {
        "sequential": Sequential,
        "ewc": EWC,
        "lwf": LwF,
        "er": ER,
        "derpp": Derpp,
        "lsf": LSF,
        "clpu": CLPU,
        "pall": PALL,
    }

    print("============================================================")
    print("[INFO] -- Experiment Configs --")
    print("       1. data & task")
    print("          dataset:      %s" % args.dataset)
    print("          n_tasks:      %d" % args.n_tasks)
    print("          # class/task: %d" % args.class_per_task)
    print("       2. training")
    print("          lr:           %5.4f" % args.lr)
    print("       3. model")
    print("          method:       %s" % args.method)
    print("          architecture: %s" % args.arch)
    print("          norm params:   %s" % args.norm_params)
    print("============================================================")

    train_datasets, test_datasets, user_requests_with_active_tasks, user_requests_without_forgotten = get_request_datasets()
    print("[INFO] finish processing data")

    path_name = os.path.join('./results', args.dataset + '_T' + str(args.n_tasks), args.arch)
    if not os.path.exists(path_name):
        os.makedirs(path_name)

    exp_name = f"seed{args.seed}_{args.method}_forget{args.n_forget}_epochs{args.n_epochs}_{args.optim}_bs{args.batch_size}_lr{args.lr}_wd{args.weight_decay}"
    exp_name += f"_mom{args.momentum}_" if args.optim == "sgd" else "_"
    if args.method == "ewc":
        exp_name += f"ewclmbd{args.ewc_lmbd}"
    elif args.method == "lwf":
        exp_name += f"lwfalpha{args.lwf_alpha}_lwftemp{args.lwf_temp}"
    elif args.method == "lsf":
        exp_name += f"lsfgamma{args.lsf_gamma}_ewclmbd{args.ewc_lmbd}_lwfalpha{args.lwf_alpha}"
        exp_name += f"_k{args.forget_iters}" if args.forget_iters else ""
    elif args.method == "er":
        exp_name += f"mem{args.mem_budget}"
        exp_name += f"_k{args.forget_iters}" if args.forget_iters else ""
    elif args.method == "derpp":
        exp_name += f"mem{args.mem_budget}_alpha{args.alpha}_beta{args.beta}"
        exp_name += f"_k{args.forget_iters}" if args.forget_iters else ""
    elif args.method == "pall":
        exp_name += f"mem{args.mem_budget}_sp{args.sparsity}_k{args.k_shot}_alpha{args.alpha}_beta{args.beta}"

    print('experiment name: ', exp_name)
    print('potential scenarios: ', [user_requests_with_active_tasks] + [user_requests_without_forgotten])

    print("[INFO] processing user requests: ", user_requests_with_active_tasks)
    model = methods_dict[args.method](args).to(args.device)
    init_model = model.state_dict()
    model.load_state_dict(init_model)
    current_stat = process_requests(args, model, train_datasets, test_datasets, user_requests_with_active_tasks)

    result = {
        'stats': current_stat,
        'user_requests_with_active_tasks': user_requests_with_active_tasks,
        'user_requests_without_forgotten': user_requests_without_forgotten,
    }

    torch.save(result, os.path.join(path_name, f"{exp_name}.pth"))


if __name__ == "__main__":
    main()
