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
from utils.args import *
from utils.ot_utils import *
from utils.eval import Meter_Unbound_Bound
from utils.early_stop import EarlyStopping
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing
import random


### Create log files only when in train mode
os.makedirs('stdouterr/', exist_ok=True)
with open(os.path.join('/apdcephfs/share_1364275/kaithgao/equidock_public/stdouterr/', banner + ".txt"), 'w') as w:
    w.write('[' + str(datetime.datetime.now()) + '] START\n')

def log(*pargs):
    with open(os.path.join('/apdcephfs/share_1364275/kaithgao/equidock_public/stdouterr/', banner + ".txt"), 'a+') as w:
        w.write('[' + str(datetime.datetime.now()) + '] ')
        w.write(" ".join(["{}".format(t) for t in pargs]))
        w.write("\n")
        pprint(*pargs)



# Ligand residue locations: a_i in R^3. Receptor: b_j in R^3
# Ligand: G_l(x) = -sigma * ln( \sum_i  exp(- ||x - a_i||^2 / sigma)  ), same for G_r(x)
# Ligand surface: x such that G_l(x) = surface_ct
# Other properties: G_l(a_i) < 0, G_l(x) = infinity if x is far from all a_i
# Intersection of ligand and receptor: points x such that G_l(x) < surface_ct && G_r(x) < surface_ct
# Intersection loss: IL = \avg_i max(0, surface_ct - G_r(a_i)) + \avg_j max(0, surface_ct - G_l(b_j))
def G_fn(protein_coords, x, sigma):
    # protein_coords: (n,3) ,  x: (m,3), output: (m,)
    e = torch.exp(- torch.sum((protein_coords.view(1, -1, 3) - x.view(-1,1,3)) ** 2, dim=2) / float(sigma) )  # (m, n)
    return - sigma * torch.log(1e-3 +  e.sum(dim=1) )

def compute_body_intersection_loss(model_ligand_coors_deform, bound_receptor_repres_nodes_loc_array, sigma, surface_ct):
    loss = torch.mean( torch.clamp(surface_ct - G_fn(bound_receptor_repres_nodes_loc_array, model_ligand_coors_deform, sigma), min=0) ) + \
           torch.mean( torch.clamp(surface_ct - G_fn(model_ligand_coors_deform, bound_receptor_repres_nodes_loc_array, sigma), min=0) )
    return loss




def run_a_generic_epoch(ep_type, args, epoch, model, data_loader, loss_fn_coors, optimizer):
    time.sleep(2)

    if ep_type == 'eval':
        model.eval()
    else:
        assert ep_type == 'train'
        model.train()

    meter = Meter_Unbound_Bound()

    avg_loss, total_loss, num_batches = 0., 0., 0

    total_loss_ligand_coors = 0.
    avg_loss_ligand_coors = 0.

    total_loss_receptor_coors = 0.
    avg_loss_receptor_coors = 0.

    total_loss_ot = 0.
    avg_loss_ot = 0.

    total_loss_intersection = 0.
    avg_loss_intersection = 0.


    num_clip = 0
    num_total_possible_clips = 0
    loader = tqdm(data_loader)

    for batch_id, batch_data in enumerate(loader):
        num_batches += 1

        if ep_type == 'train':
            optimizer.zero_grad()

        batch_ligand_graph, batch_receptor_graph, \
        bound_ligand_repres_nodes_loc_array_list, bound_receptor_repres_nodes_loc_array_list, \
        pocket_coors_ligand_list, pocket_coors_receptor_list = batch_data
        # batch_hetero_graph, \
        # bound_ligand_repres_nodes_loc_array_list, bound_receptor_repres_nodes_loc_array_list, \
        # pocket_coors_ligand_list, pocket_coors_receptor_list = batch_data

        batch_ligand_graph = batch_ligand_graph.to(args['device'])
        batch_receptor_graph = batch_receptor_graph.to(args['device'])


        ######## RUN MODEL ##############
        model_ligand_coors_deform_list, \
        model_keypts_ligand_list, model_keypts_receptor_list, \
        _, _,TEMP_1,TEMP_2  = model(batch_ligand_graph, batch_receptor_graph, epoch=epoch)
        ################################



        # Compute MSE loss for each protein individually, then average over the minibatch.
        batch_ligand_coors_loss = torch.zeros([]).to(args['device'])
        batch_receptor_coors_loss = torch.zeros([]).to(args['device'])
        batch_ot_loss = torch.zeros([]).to(args['device'])
        batch_intersection_loss = torch.zeros([]).to(args['device'])

        assert len(pocket_coors_ligand_list) == len(model_ligand_coors_deform_list)
        assert len(bound_ligand_repres_nodes_loc_array_list) == len(model_ligand_coors_deform_list)

        for i in range(len(model_ligand_coors_deform_list)):
            ## Compute average MSE loss (which is 3 times smaller than average squared RMSD)
            batch_ligand_coors_loss = batch_ligand_coors_loss + loss_fn_coors(model_ligand_coors_deform_list[i],
                                                                              bound_ligand_repres_nodes_loc_array_list[i].to(args['device']))

            # Compute the OT loss for the binding pocket:
            ligand_pocket_coors = pocket_coors_ligand_list[i].to(args['device'])  ##  (N, 3), N = num pocket nodes
            receptor_pocket_coors = pocket_coors_receptor_list[i].to(args['device'])  ##  (N, 3), N = num pocket nodes

            ligand_keypts_coors = model_keypts_ligand_list[i]  ##  (K, 3), K = num keypoints
            receptor_keypts_coors = model_keypts_receptor_list[i]  ##  (K, 3), K = num keypoints

            ## (N, K) cost matrix
            cost_mat_ligand = compute_sq_dist_mat(ligand_pocket_coors, ligand_keypts_coors)
            cost_mat_receptor = compute_sq_dist_mat(receptor_pocket_coors, receptor_keypts_coors)

            ot_dist, _ = compute_ot_emd(cost_mat_ligand + cost_mat_receptor, args['device'])
            batch_ot_loss = batch_ot_loss + ot_dist

            batch_intersection_loss = batch_intersection_loss + compute_body_intersection_loss(
                model_ligand_coors_deform_list[i], bound_receptor_repres_nodes_loc_array_list[i].to(args['device']),
                args['intersection_sigma'], args['intersection_surface_ct'])

            ### Add new stats to the meter
            if ep_type != 'train' or random.random() < 0.1:
                meter.update_rmsd(model_ligand_coors_deform_list[i],
                                  bound_receptor_repres_nodes_loc_array_list[i],
                                  bound_ligand_repres_nodes_loc_array_list[i],
                                  bound_receptor_repres_nodes_loc_array_list[i])


        batch_ligand_coors_loss = batch_ligand_coors_loss / float(len(model_ligand_coors_deform_list))
        batch_receptor_coors_loss = batch_receptor_coors_loss / float(len(model_ligand_coors_deform_list))
        batch_ot_loss = batch_ot_loss / float(len(model_ligand_coors_deform_list))
        batch_intersection_loss = batch_intersection_loss  / float(len(model_ligand_coors_deform_list))

        loss_coors = batch_ligand_coors_loss + batch_receptor_coors_loss

        loss = loss_coors + args['pocket_ot_loss_weight'] * batch_ot_loss + args['intersection_loss_weight'] * batch_intersection_loss

        #########
        if ep_type == 'train':
            loss.backward()

            clip = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args['clip'], norm_type=2)
            if clip > args['clip']:
                # gradient exploded
                # if clip > args['clip'] * 100 and num_batches > 1:
                    # log(f"Gradient Exploded: {clip}")
                    # optimizer.zero_grad()
                num_clip += 1
            num_total_possible_clips += 1

            optimizer.step()

            for name, param in model.named_parameters():
                if param.norm() > 500.:
                    log('    PARAM: ', name, ' --> norm = ', param.norm(), ' --> grad = ', param.grad.norm())
        ###########

        total_loss += loss.detach()
        total_loss_ligand_coors += batch_ligand_coors_loss.detach()
        total_loss_receptor_coors += batch_receptor_coors_loss.detach()
        total_loss_ot += batch_ot_loss.detach()
        total_loss_intersection += batch_intersection_loss.detach()

        if batch_id % args['log_every'] == args['log_every']-1:
            log('batch {:.0f}% || Loss {:.6f}'.format((100. * batch_id) / len(data_loader), loss.item()))


    if num_batches != 0:
        avg_loss = total_loss / num_batches
        avg_loss_ligand_coors = total_loss_ligand_coors / num_batches
        avg_loss_receptor_coors = total_loss_receptor_coors / num_batches
        avg_loss_ot = total_loss_ot / num_batches
        avg_loss_intersection = total_loss_intersection / num_batches

    ligand_rmsd_mean, receptor_rmsd_mean, complex_rmsd_mean = meter.summarize(reduction_rmsd='mean')
    ligand_rmsd_median, receptor_rmsd_median, complex_rmsd_median = meter.summarize(reduction_rmsd='median')


    #########
    if ep_type == 'train':
        log('[TRAIN] -->  total loss {:.4f} || lrmsd loss {:.4f} || OT loss {:.4f} || INTERSC loss {:.4f} || lr {:.7f} || num grad clips {:d}/{:d}'.format(
            avg_loss, avg_loss_ligand_coors, avg_loss_ot, avg_loss_intersection, optimizer.param_groups[0]['lr'], num_clip, num_total_possible_clips))

        pretty_print_stats('TRAIN', epoch, args['num_epochs'],
                           complex_rmsd_mean, complex_rmsd_median,
                           ligand_rmsd_mean, ligand_rmsd_median,
                           receptor_rmsd_mean, receptor_rmsd_median,
                           avg_loss, avg_loss_ligand_coors, avg_loss_receptor_coors, avg_loss_ot, avg_loss_intersection,
                           log)

    percentage_clips = float(num_clip) / float(1 + num_total_possible_clips)

    return complex_rmsd_mean, complex_rmsd_median, \
           ligand_rmsd_mean, ligand_rmsd_median,\
           receptor_rmsd_mean, receptor_rmsd_median,\
           avg_loss.item(), avg_loss_ligand_coors.item(),\
           avg_loss_receptor_coors.item(),\
           avg_loss_ot.item(), avg_loss_intersection.item(), percentage_clips



def run_a_train_epoch(args, epoch, model, data_loader, loss_fn_coors, optimizer):
    return run_a_generic_epoch('train', args, epoch, model, data_loader, loss_fn_coors, optimizer)


def run_an_eval_epoch(args, model, data_loader, loss_fn_coors=None):
    with torch.no_grad():
        complex_rmsd_mean, complex_rmsd_median, \
        ligand_rmsd_mean, ligand_rmsd_median, \
        receptor_rmsd_mean, receptor_rmsd_median, \
        avg_loss, avg_loss_ligand_coors, \
        avg_loss_receptor_coors,\
        avg_loss_ot, avg_loss_intersection, percentage_clips = \
            run_a_generic_epoch('eval', args=args, epoch=-1, model=model, data_loader=data_loader,
                                loss_fn_coors=loss_fn_coors, optimizer=None)

    return complex_rmsd_mean, complex_rmsd_median, \
           ligand_rmsd_mean, ligand_rmsd_median, \
           receptor_rmsd_mean, receptor_rmsd_median, \
           avg_loss, avg_loss_ligand_coors, \
           avg_loss_receptor_coors, \
           avg_loss_ot, avg_loss_intersection


def main(args):

    create_dir(args['checkpoint_dir'])
    create_dir(args['tb_log_dir'])

    tb_logger = SummaryWriter(log_dir=args['tb_log_dir'])


    # TODO: hack to init input_edge_feats_dim
    args['data'] = 'db5'
    args['data_fraction'] = 1.
    args['cache_path'] = '/apdcephfs/share_1364275/kaithgao/equidock_public/cache/' + args['data'] + '_' + args['graph_nodes'] + '_maxneighbor_' + \
                         str(args['graph_max_neighbor']) + '_cutoff_' + str(args['graph_cutoff']) + \
                         '_pocketCut_' + str(args['pocket_cutoff']) + '/cv_' + str(args['split'])
    get_dataloader(args, log)
    ### End of hack


    checkpoint = torch.load('/apdcephfs/share_1364275/kaithgao/equidock_public/checkpts/EQUIDOCK__drp_0.0#Wdec_0.0001#ITS_lw_10.0#Hdim_64#Nlay_5#shrdLay_F#SURFfs_F#ln_LN#lnX_0#Hnrm_0#NattH_50#skH_0.5#xConnI_0.0#LkySl_0.01#pokOTw_1.0#divXdist_F#20:41:18/db5_model_best.pth', map_location=args['device'])
    model = create_model(args, log)
    model.load_state_dict(checkpoint['state_dict'])
    param_count(model, log)
    model = model.to(args['device'])
    model.eval()

    if torch.cuda.is_available():
        model.cuda()


    # if not args['toy']:

    #     ## Train DIPS first
        # args['pocket_cutoff'] = 8.
        args['lr'] = 2e-4
        # args['data'] = 'dips'
        args['checkpoint_filename'] = os.path.join(args['checkpoint_dir'], args['data'] + '_model_best.pth')
        args['data_fraction'] = 1.
        args['patience'] = 100
        args['warmup'] = 1.
        args['split'] = 0
        model = test(args, tb_logger, model, nn.MSELoss(reduction='mean'))


    # args['pocket_cutoff'] = 8.
    # args['lr'] = 1e-4
    # args['data'] = 'dips'
    # args['checkpoint_filename'] = os.path.join(args['checkpoint_dir'], args['data'] + '_model_best.pth')
    # args['data_fraction'] = 1.
    # args['patience'] = 500
    # args['warmup'] = 1.
    # args['split'] = 0
    # model = train(args, tb_logger, model, nn.MSELoss(reduction='mean'))



def test(args, tb_logger, model, loss_fn_coors):
    tb_banner = args['data'] + '_'

    args['cache_path'] = '/apdcephfs/share_1364275/kaithgao/equidock_public/cache/' + args['data'] + '_' + args['graph_nodes'] + '_maxneighbor_' + \
                         str(args['graph_max_neighbor']) + '_cutoff_' + str(args['graph_cutoff']) + \
                         '_pocketCut_' + str(args['pocket_cutoff']) + '/'
    args['cache_path'] = os.path.join(args['cache_path'], 'cv_' + str(args['split']))


    train_loader, val_loader, test_loader = get_dataloader(args, log)

    stopper = EarlyStopping(mode='lower', patience=args['patience'], filename=args['checkpoint_filename'], log=log)

    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['w_decay'])

    lr_scheduler = get_scheduler(optimizer, args)

    # Start Epoch
    best_epoch = 0

    best_val_rmsd_median = float('inf')
    corr_val_rmsd_mean = float('inf')

    
    # Load the latest checkpoint for final eval
    # model, optimizer, args2, epoch = stopper.load_checkpoint(model, optimizer)
    # for k in args2.keys():
    #     if k not in ['device', 'debug', 'worker', 'n_jobs', 'toy']:
    #         assert args[k] == args2[k]

    
    test_complex_rmsd_mean, test_complex_rmsd_median, \
    test_ligand_rmsd_mean, test_ligand_rmsd_median, \
    test_receptor_rmsd_mean, test_receptor_rmsd_median, \
    test_avg_loss, test_avg_loss_ligand_coors, \
    test_avg_loss_receptor_coors, \
    test_avg_loss_ot, test_avg_loss_intersection = run_an_eval_epoch(args, model, test_loader, loss_fn_coors=loss_fn_coors)
    pretty_print_stats('FINAL TEST for ' + args['data'], -1, args['num_epochs'],
                       test_complex_rmsd_mean, test_complex_rmsd_median,
                       test_ligand_rmsd_mean, test_ligand_rmsd_median,
                       test_receptor_rmsd_mean, test_receptor_rmsd_median,
                       test_avg_loss, test_avg_loss_ligand_coors, test_avg_loss_receptor_coors,
                       test_avg_loss_ot, test_avg_loss_intersection, log)

    return model




if __name__ == "__main__":
    main(args)