import numpy as np
import torch
import os
from dataset import Dataset, collate_fn
from utils.utils import compute_auc, compute_accuracy, data_split, batch_accuracy
from model import MAMLModel
from policy import MyModel
from copy import deepcopy
from utils.configuration import create_parser, initialize_seeds
import time
import os
import json

NEPTUNE_API_TOKEN ="YOUR TOKEN"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_val_score, best_test_score = 0, 0
best_val_auc, best_test_auc = 0, 0
best_epoch = -1
def find_nonzeros(tensor):
    nz_idx = tensor.nonzero()  
    nz_val = tensor[nz_idx[:,0], nz_idx[:,1]]  
    return nz_idx, nz_val

def clone_meta_params(batch):
    return [meta_params[0].expand(len(batch['input_labels']),  -1).clone(
    )]

def inner_algo(batch, config, new_params, create_graph=False):

    for _ in range(params.inner_loop):
        config['meta_param'] = new_params[0]
        res = model(batch, config)
        loss = res['train_loss']
        grads = torch.autograd.grad(
            loss, new_params, create_graph=create_graph)
        new_params = [(new_params[i] - params.inner_lr*grads[i])
                      for i in range(len(new_params))]
        del grads
    config['meta_param'] = new_params[0]
    return

def DARTS_algo(batch, config, new_params, create_graph=False):
    #
    config['meta_param'] = new_params[0]
    res = model(batch, config)
    loss_train = res['train_loss']
    grads_train = torch.autograd.grad(
        loss_train, new_params, create_graph=create_graph)
    tmp_params = [(new_params[i] - params.inner_lr*grads_train[i])
                    for i in range(len(new_params))]   
    del grads_train
    config['meta_param'] = tmp_params[0]
    res = model(batch,config)
    loss_val_tmp = res['loss']
    grads_val_tmp = torch.autograd.grad(
        loss_val_tmp, tmp_params, create_graph=create_graph)
    l2_norm = torch.norm(grads_val_tmp[0], p=2)
    epl = l2_norm
    params_plus = [(new_params[i] + epl*grads_val_tmp[i])
                    for i in range(len(tmp_params))] 
    params_sub =  [(new_params[i] - epl*grads_val_tmp[i])
                    for i in range(len(tmp_params))] 
    del grads_val_tmp
    config['meta_param'] = params_plus[0]
    res = model(batch,config)
    loss_train_plus = res['train_loss']
    config['meta_param'] = params_sub[0]
    res = model(batch,config)
    loss_train_sub = res['train_loss']
    config['meta_param'] = new_params[0]
    return  loss_val_tmp,loss_train_plus,loss_train_sub,epl
     
def pick_DARTS_samples(batch, config):
    new_params = clone_meta_params(batch)

    if config['mode'] == 'train':
        model.eval()
    env_states = model.reset(batch)
    action_mask, train_mask = env_states['action_mask'], env_states['train_mask']
    i = 0
    while i <= params.n_query:
        with torch.no_grad():
            state = model.step(env_states)
            train_mask = env_states['train_mask']
        feature = model.difficulty_irt_old()
        if config['mode'] == 'train':
            train_mask_sample, actions = mymodel.policy(state, action_mask,feature)
        else:
            with torch.no_grad():
                train_mask_sample, actions = mymodel.policy(state, action_mask,feature)
        if params.option_num != 0:
            for action in actions:
                action_mask[range(len(action_mask)), action] = 0
        else:
            action_mask[range(len(action_mask)), actions] = 0

        # env state train mask should be detached
        env_states['train_mask']=train_mask + \
            train_mask_sample.data
        env_states['action_mask'] = action_mask
        if config['mode'] == 'train':
            # loss computation train mask should flow gradient
            config['train_mask'] = train_mask_sample+train_mask
            l_val_t,l_train_p,l_train_s,epl= DARTS_algo(batch, config, new_params, create_graph=True)
            
            mymodel.update(l_val_t,l_train_p,l_train_s,epl,params.inner_lr)
            inner_algo(batch,config,new_params)
            
        i = i + params.option_num +1
    config['train_mask'] = env_states['train_mask']
    return 

def run_unbiased_DARTS(batch, config):
    new_params = clone_meta_params(batch)

    if config['mode'] == 'train':
        model.eval()
    pick_DARTS_samples(batch, config)
    optimizer.zero_grad()
    meta_params_optimizer.zero_grad()
    inner_algo(batch, config, new_params)
    if config['mode'] == 'train':
        model.train()
        optimizer.zero_grad()
        res = model(batch, config)
        loss = res['loss']
        loss.backward()
        optimizer.step()
        meta_params_optimizer.step()
        ####
    else:
        with torch.no_grad():
            res = model(batch, config)
    return res['output']

def run_random(batch, config):
    new_params = clone_meta_params(batch)
    meta_params_optimizer.zero_grad()
    if config['mode'] == 'train':
        optimizer.zero_grad()
    ###
    config['available_mask'] = batch['input_mask'].to(device).clone()
    config['train_mask'] = torch.zeros(
        len(batch['input_mask']), params.n_question).long().to(device)

    # Random pick once
    config['meta_param'] = new_params[0]
    if sampling == 'random':
        model.pick_sample('random', config)
        inner_algo(batch, config, new_params)
    if sampling == 'active':
        for _ in range(params.n_query):
            model.pick_sample('active', config)
            inner_algo(batch, config, new_params)

    if config['mode'] == 'train':
        res = model(batch, config)
        loss = res['loss']
        loss.backward()
        optimizer.step()
        meta_params_optimizer.step()
        return
    else:
        with torch.no_grad():
            res = model(batch, config)
        output = res['output']
        return output

def train_model():
    global best_val_auc, best_test_auc, best_val_score, best_test_score, best_epoch
    config['mode'] = 'train'
    config['epoch'] = epoch
    model.train()
    N = [idx for idx in range(100, 100+params.repeat)]
    for batch in train_loader:
        # Select RL Actions, save in config
        if sampling == 'DARTS':
            run_unbiased_DARTS(batch,config)
        else:
            run_random(batch, config)
    # Validation
    val_scores, val_aucs = [], []
    test_scores, test_aucs = [], []
    for idx in N:
        _, auc, acc= test_model(id_=idx, split='val')
        val_scores.append(acc)
        val_aucs.append(auc)
    val_score = sum(val_scores)/(len(N)+1e-20)
    val_auc = sum(val_aucs)/(len(N)+1e-20)

    if best_val_score < val_score:
        best_epoch = epoch
        best_val_score = val_score
        best_val_auc = val_auc
        # Run on test set
        for idx in N:
            _, auc, acc = test_model(id_=idx, split='test')
            test_scores.append(acc)
            test_aucs.append(auc)
        best_test_score = sum(test_scores)/(len(N)+1e-20)
        best_test_auc = sum(test_aucs)/(len(N)+1e-20)
    #   
    print('Test_Epoch: {}; val_scores: {}; val_aucs: {}; test_scores: {}; test_aucs: {}'.format(
       epoch, val_scores, val_aucs, test_scores, test_aucs))
    elapsed = time.time() - start_time
    if params.neptune:
        neptune_exp["Valid Accuracy"].append(val_score)
        neptune_exp["Best Test Accuracy"].append(best_test_score)
        neptune_exp["Best Test Auc"].append( best_test_auc)
        neptune_exp["Best Valid Accuracy"].append( best_val_score)
        neptune_exp["Best Valid Auc"].append(best_val_auc)
        neptune_exp["Best Epoch"].append(best_epoch)
        neptune_exp["Epoch"].append(epoch)
        neptune_exp['Time'].append(elapsed)

def test_model(id_, split='val'):
    model.eval()
    config['mode'] = 'test'
    if split == 'val':
        valid_dataset.seed = id_
    elif split == 'test':
        test_dataset.seed = id_
    loader = torch.utils.data.DataLoader(
        valid_dataset if split == 'val' else test_dataset, collate_fn=collate_fn, batch_size=params.test_batch_size, num_workers=num_workers, shuffle=False, drop_last=False)

    total_loss, all_preds, all_targets = 0., [], []
    n_batch = 0
    for batch in loader:
        if sampling == 'DARTS':
            output= run_unbiased_DARTS(batch, config)
        else:
            output = run_random(batch, config)
        target = batch['output_labels'].float().numpy()
        mask = batch['output_mask'].numpy() == 1
        all_preds.append(output[mask])
        all_targets.append(target[mask])
        n_batch += 1

    all_pred = np.concatenate(all_preds, axis=0)
    all_target = np.concatenate(all_targets, axis=0)
    auc = compute_auc(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)
    return total_loss/n_batch, auc, accuracy

if __name__ == "__main__":
    params = create_parser()
    print(params)

    if params.use_cuda:
        assert device.type == 'cuda', 'no gpu found!'

    if params.neptune:
        import neptune
        print('success import ')
        project = "Your project name"
        neptune_exp = neptune.init_run(project=project,
                     api_token=NEPTUNE_API_TOKEN)
    config = {}
    initialize_seeds(params.seed)
    #
    base, sampling = params.model.split('-')[0], params.model.split('-')[-1]
    if base == 'biirt':
        model = MAMLModel(sampling=sampling, n_query=params.n_query,
                          n_question=params.n_question, question_dim=1,tp = 'irt').to(device)
        meta_params = [torch.zeros(1, 1, device=device, requires_grad=True)]
    if base == 'binn':
        concept_name = params.dataset +'_concept_map.json'
        with open(concept_name, 'r') as file:
            concepts = json.load(file)
        num_concepts = params.concept_num
        concepts_emb = [[0.] * num_concepts for i in range(params.n_question)]
        if params.dataset=='exam':
            for i in range(1,params.n_question):
                for concept in concepts[str(i)]:
                    concepts_emb[i][concept] = 1.0   
        else:
            for i in range(params.n_question):
                for concept in concepts[str(i)]:
                    concepts_emb[i][concept] = 1.0
        concepts_emb = torch.tensor(concepts_emb, dtype=torch.float32).to(device)
        model = MAMLModel(sampling=sampling, n_query=params.n_query,
                          n_question=params.n_question, question_dim=params.question_dim,tp ='ncd',emb=concepts_emb).to(device)
        meta_params = [torch.zeros((1, num_concepts), device=device, requires_grad=True)]
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params.lr, weight_decay=1e-8)

    meta_params_optimizer = torch.optim.SGD(
        meta_params, lr=params.meta_lr, weight_decay=2e-6, momentum=0.9)
    if params.neptune:
        neptune_exp["model_summary"].append(repr(model))
    print(model)
    #
    if sampling == 'DARTS':
        betas = (0.9, 0.999)
        mymodel = MyModel(params.n_question, params.n_question,
                    params.policy_lr, betas,params.option_num)
        if params.neptune:
            neptune_exp["DARTS_model_summary"].append(repr(mymodel.policy))
    #
    data_path = os.path.normpath('data/train_task_'+params.dataset+'.json')
    train_data, valid_data, test_data = data_split(
        data_path, params.fold,  params.seed)
    train_dataset, valid_dataset, test_dataset = Dataset(
        train_data), Dataset(valid_data), Dataset(test_data)
    #
    num_workers = 3
    collate_fn = collate_fn(params.n_question)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, collate_fn=collate_fn, batch_size=params.train_batch_size, num_workers=num_workers, shuffle=True, drop_last=True)
    start_time = time.time()
    for epoch in range(params.n_epoch):
        train_model()
        if epoch >= (best_epoch+params.wait):
            print(best_epoch,best_test_auc,best_test_score)
            break
