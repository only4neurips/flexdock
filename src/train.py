# -*- coding: utf-8 -*-
#
import os
import time
import sys
# sys.path.insert(0, '../')
# sys.path.insert(0, '../src')
# sys.path.insert(0, './src')
# sys.path.insert(0, './')
# # sys.path.append(os.path.dirname(sys.path[0]))
# sys.path.insert(0, '/apdcephfs/share_1364275/kaithgao/equidock_public/src/')
from utils.io import create_dir
os.environ['DGLBACKEND'] = 'pytorch'
from datetime import datetime as dt
from utils.train_utils import *
from utils.protein_utils import rigid_transform_Kabsch_3D
from utils.args import *
from utils.ot_utils import *
from utils.eval import Meter_Unbound_Bound
from utils.early_stop import EarlyStopping
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing
import random
from sklearn.metrics import accuracy_score
import warnings
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

os.makedirs('./my_log/', exist_ok=True)
with open(os.path.join('./my_log/', banner + ".txt"), 'w') as w:
    w.write('[' + str(datetime.datetime.now()) + '] START\n')

def log(*pargs):
    with open(os.path.join('./my_log/', banner + ".txt"), 'a+') as w:
        w.write('[' + str(datetime.datetime.now()) + '] ')
        w.write(" ".join(["{}".format(t) for t in pargs]))
        w.write("\n")
        pprint(*pargs)




def G_fn(protein_coords, x, sigma):
    # protein_coords: (n,3) ,  x: (m,3), output: (m,)
    e = torch.exp(- torch.sum((protein_coords.view(1, -1, 3) - x.view(-1,1,3)) ** 2, dim=2) / float(sigma) )  # (m, n)
    return - sigma * torch.log(1e-3 +  e.sum(dim=1) )


def rmsd_self(complex_coors_pred,complex_coors_true, device):
    R,b = rigid_transform_Kabsch_3D(complex_coors_pred.T, complex_coors_true.T, device)
    complex_coors_pred_aligned = ( (R @ complex_coors_pred.T) + b ).T
    complex_rmsd = torch.sqrt(torch.mean(torch.sum( (complex_coors_pred_aligned - complex_coors_true) ** 2, axis=1)))
    return complex_rmsd



def run_a_generic_epoch(ep_type, args, epoch, model, data_loader, loss_fn_coors, optimizer):
    time.sleep(2)

    if ep_type == 'eval':
        model.eval()
    else:
        assert ep_type == 'train'
        model.train()

    meter = Meter_Unbound_Bound()

    avg_loss, total_loss, num_batches, total_acc, total_auc, total_rmsd = 0., 0., 0, 0, 0, 0.

    loader = tqdm(data_loader)
    rmsd_wo_nan = []

    for batch_id, batch_data in enumerate(loader):
        num_batches += 1

        if ep_type == 'train':
            optimizer.zero_grad()

        batch_ligand_graph, batch_receptor_graph, \
        bound_ligand_repres_nodes_loc_array_list, bound_receptor_repres_nodes_loc_array_list, \
        pocket_coors_ligand_list, pocket_coors_receptor_list, train_tuple, train_label_tuple = batch_data
        # batch_hetero_graph, \
        # bound_ligand_repres_nodes_loc_array_list, bound_receptor_repres_nodes_loc_array_list, \
        # pocket_coors_ligand_list, pocket_coors_receptor_list = batch_data
        # ind_a = round(train_tuple[0].shape[1]/5.5)
        # if ep_type == 'train': 
        #     train_tuple[0] = train_tuple[0][:,:ind_a]
        #     train_label_tuple[0] = train_label_tuple[0][:ind_a]
        batch_ligand_graph = batch_ligand_graph.to(args['device'])
        batch_receptor_graph = batch_receptor_graph.to(args['device'])


        ######## RUN MODEL ##############
        TEMP_1, TEMP_2, pre_interface_list, x_1_final_list, x_2_final_list = model(batch_ligand_graph, batch_receptor_graph, train_tuple,train_label_tuple, epoch=epoch)
        ################################

        # print(TEMP_1)
        # print(TEMP_2)

        
        batch_stable_loss = torch.max(torch.abs(torch.diff(TEMP_2))).to(args['device'])
        batch_interface_loss = torch.zeros([]).to(args['device'])
        batch_acc = torch.zeros([]).to(args['device'])
        batch_auc = torch.zeros([]).to(args['device'])
        sc=torch.tensor([1,10]).to(args['device'])
        batch_rmsd_loss = torch.zeros([]).to(args['device'])
        int_criterion = nn.BCELoss(weight=sc)
        unbatch_graph_1_list = dgl.unbatch(batch_ligand_graph)

        for i in range(len(pre_interface_list)):

            target = torch.cat((1-train_label_tuple[i].unsqueeze(1).cuda(),train_label_tuple[i].unsqueeze(1).cuda()),dim=1)
            
            if ep_type == 'train':
                pos_num = (target[:,1].mean()*pre_interface_list[i].shape[0]).long()
                if target[:,1].mean() > 0:
                    sample_pos_ind = random.sample(range(pos_num), (pos_num/2).long())
                    sample_neg_ind = random.sample(range(pos_num,pre_interface_list[i].shape[0]), (pos_num/10).long())
                    sample_all_pre = torch.cat((pre_interface_list[i][sample_pos_ind],pre_interface_list[i][sample_neg_ind]),dim=0)
                    sample_all_tar = torch.cat((target[sample_pos_ind],target[sample_neg_ind]),dim=0)
                    batch_interface_loss = batch_interface_loss + \
                        int_criterion(sample_all_pre, sample_all_tar)
                else:
                    batch_interface_loss = batch_interface_loss + \
                        int_criterion(pre_interface_list[i], target)
            else:
                    batch_interface_loss = batch_interface_loss + \
                        int_criterion(pre_interface_list[i], target)

            batch_acc += accuracy_score(np.array(train_label_tuple[i].cpu()),np.array(torch.argmax(pre_interface_list[i],1).cpu()))
            batch_auc += roc_auc_score(np.array(train_label_tuple[i].detach().cpu()),np.array(pre_interface_list[i][:,1].detach().cpu()))

            gt_coors = torch.cat((unbatch_graph_1_list[i].ndata['x'],batch_receptor_graph.ndata['x']),dim=0)
            pre_coors = torch.cat((x_1_final_list[i],batch_receptor_graph.ndata['x']),dim=0)
            batch_rmsd_loss += rmsd_self(gt_coors, pre_coors, args['device'])

        if_loss = batch_interface_loss/len(pre_interface_list)
        rmsd_loss = batch_rmsd_loss/len(pre_interface_list)
        stable_loss = batch_stable_loss
        rmsd_wo_nan.append(rmsd_self(gt_coors, pre_coors, args['device']))
        #########
        if args['stage_2'] == True:
            loss = args['gamma'] * stable_loss + if_loss + rmsd_loss
        else:
            loss = rmsd_loss

        if ep_type == 'train':
            loss.backward()
            optimizer.step()

        if ep_type == 'train':
            num = 205
        else:
            num = 25
        
        total_acc += batch_acc
        total_loss += batch_interface_loss.detach()
        total_auc += batch_auc
        total_rmsd += batch_rmsd_loss
        if ep_type == 'eval':
            if total_auc/num > 0.85:
                args['stage_2'] = True

    return total_loss/num, total_acc/num, total_auc/num, total_rmsd/num, rmsd_wo_nan



def run_a_train_epoch(args, epoch, model, data_loader, loss_fn_coors, optimizer):
    return run_a_generic_epoch('train', args, epoch, model, data_loader, loss_fn_coors, optimizer)


def run_an_eval_epoch(args, model, data_loader, loss_fn_coors=None):
    with torch.no_grad():
        loss = \
            run_a_generic_epoch('eval', args=args, epoch=-1, model=model, data_loader=data_loader,
                                loss_fn_coors=loss_fn_coors, optimizer=None)

    return loss


def main(args):

    create_dir(args['checkpoint_dir'])
    create_dir(args['tb_log_dir'])

    tb_logger = SummaryWriter(log_dir=args['tb_log_dir'])

    if args['toy'] == True:
        args['data'] = 'db5'
        args['finetune'] = False
    
    args['data_fraction'] = 1.
    args['cache_path'] = './cache/' + args['data'] + '_' + args['graph_nodes'] + '_maxneighbor_' + \
                         str(args['graph_max_neighbor']) + '_cutoff_' + str(args['graph_cutoff']) + \
                         '_pocketCut_' + str(args['pocket_cutoff']) + '/cv_' + str(args['split'])
    get_dataloader(args, log)
    model = create_model(args, log)
    if torch.cuda.is_available():
        model.cuda()
    param_count(model, log, print_model=False)
    if args['finetune'] == True:
        check_path = "./checkpts/FLEXDOCK#20:40:48/dips_model_best.pth"
        model = torch.load(check_path)

    args['checkpoint_filename'] = os.path.join(args['checkpoint_dir'], args['data'] + '_model_best.pth')
    args['data_fraction'] = 1.
    args['patience'] = 100
    args['warmup'] = 1.
    args['split'] = 0
    train(args, tb_logger, model, nn.MSELoss(reduction='mean'))




def train(args, tb_logger, model, loss_fn_coors):
    tb_banner = args['data'] + '_'


    args['cache_path'] = '/apdcephfs/share_1364275/kaithgao/flexdock_git/flexdock_0425/cache/' + args['data'] + '_' + args['graph_nodes'] + '_maxneighbor_' + \
                         str(args['graph_max_neighbor']) + '_cutoff_' + str(args['graph_cutoff']) + \
                         '_pocketCut_' + str(args['pocket_cutoff']) + '/cv_' + str(args['split'])


    train_loader, val_loader, test_loader = get_dataloader(args, log)

    stopper = EarlyStopping(mode='lower', patience=args['patience'], filename=args['checkpoint_filename'], log=log)

    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['w_decay'])

    lr_scheduler = get_scheduler(optimizer, args)

    # Start Epoch
    best_epoch = 0

    best_val_rmsd = float('inf')

    for epoch in range(10000):

        log('+' * 100)
        log('Model name ===> ', banner)
        log('epoch = ',epoch, 'stage 2 = ', args['stage_2'])

        epoch_start = dt.now()
        # Train
        train_loss, train_acc, train_auc, train_rmsd, train_rmsd_nan = run_a_train_epoch(args, epoch, model, train_loader, loss_fn_coors=loss_fn_coors, optimizer=optimizer)


        val_loss, val_acc, val_auc, val_rmsd, val_rmsd_nan = run_an_eval_epoch(args, model, val_loader, loss_fn_coors=loss_fn_coors)

        test_loss, test_acc, test_auc, test_rmsd, test_rmsd_nan = run_an_eval_epoch(args, model, test_loader, loss_fn_coors=loss_fn_coors)

        lr_scheduler.step(train_auc)

        log(np.array(train_loss.cpu()))
        log('train_acc=',train_acc,'train_auc=',train_auc) 
        log('val_acc=',val_acc,'val_auc=',val_auc)
        log('test_acc=',test_acc,'test_auc=',test_auc)
        log('train_rmsd=', train_rmsd)
        # log('train_rmsd=', np.array(torch.tensor(train_rmsd_nan).cpu()))
        # log('train_rmsd=', np.array(torch.tensor(train_rmsd_nan).cpu()))
        log('val_rmsd=', val_rmsd)
        log('val_rmsd=', np.array(torch.tensor(val_rmsd_nan).cpu()))
        log('test_rmsd=', test_rmsd)
        log('test_rmsd=', np.array(torch.tensor(test_rmsd_nan).cpu()))
        
        if val_rmsd < best_val_rmsd * 0.98: ## We do this to avoid "pure luck"
            # Best validation so far
            best_val_rmsd = val_rmsd
            best_epoch = epoch

        early_stop = stopper.step(best_val_rmsd, model, optimizer, args, epoch, True)
        log('[BEST SO FAR] ', args['data'],
            '|| At best epoch {} we have: best_val_rmsd_mean {:.4f}, '
            '|| Current val rmsd_median {:.4f}  '
            '|| Train time: {}\n'.
            format(best_epoch, best_val_rmsd,
                   val_rmsd,
                   dt.now() - epoch_start))





if __name__ == "__main__":
    main(args)