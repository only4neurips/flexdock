import math
import sys
from torch.nn import init
import numpy as np
from torch.nn import Parameter
import dgl
import torch
from torch import nn
from dgl import function as fn
from utils.graph_norm import GraphNorm
import sys
import random
from model.Bernpro import Bern_prop
import torch.nn.functional as F
import os
from sklearn.ensemble import IsolationForest
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
def get_non_lin(type, negative_slope):
    if type == 'swish':
        return nn.SiLU()
    else:
        assert type == 'lkyrelu'
        return nn.LeakyReLU(negative_slope=negative_slope)


def get_layer_norm(layer_norm_type, dim):
    if layer_norm_type == 'BN':
        return nn.BatchNorm1d(dim)
    elif layer_norm_type == 'LN':
        return nn.LayerNorm(dim)
    else:
        return nn.Identity()


def get_final_h_layer_norm(layer_norm_type, dim):
    if layer_norm_type == 'BN':
        return nn.BatchNorm1d(dim)
    elif layer_norm_type == 'LN':
        return nn.LayerNorm(dim)
    elif layer_norm_type == 'GN':
        return GraphNorm(dim)
    else:
        assert layer_norm_type == '0'
        return nn.Identity()


def apply_final_h_layer_norm(g, h, node_type, norm_type, norm_layer):
    if norm_type == 'GN':
        return norm_layer(g, h, node_type)
    return norm_layer(h)





def get_mask(ligand_batch_num_nodes, receptor_batch_num_nodes, device):
    rows = sum(ligand_batch_num_nodes)
    cols = sum(receptor_batch_num_nodes)
    mask = torch.zeros(rows, cols).to(device)
    partial_l = 0
    partial_r = 0
    for l_n, r_n in zip(ligand_batch_num_nodes, receptor_batch_num_nodes):
        mask[partial_l: partial_l + l_n, partial_r: partial_r + r_n] = 1
        partial_l = partial_l + l_n
        partial_r = partial_r + r_n
    return mask



class SEGCN_Layer(nn.Module):
    def __init__(self, args):

        super(SEGCN_Layer, self).__init__()
        hidden = args['h_dim']
        coe = 10
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        # EDGES
        self.all_sigmas_dist = [100 ** x for x in range(coe)]
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(coe,hidden),
            nn.Dropout(args['dropout']),
            get_non_lin('lkyrelu', 0.02),
            get_layer_norm('BN', hidden),
            nn.Linear(hidden, 1),
        )
        self.x_mlp = nn.Sequential(
            nn.Linear(hidden*2,hidden*2),
            nn.Dropout(args['dropout']),
            get_non_lin('lkyrelu', 0.02),
            get_layer_norm('BN', hidden*2),
            nn.Linear(hidden*2, 1),
        )
        self.fea_mlp_1 = nn.Sequential(
            nn.Linear(hidden,hidden),
            nn.Dropout(args['dropout']),
            get_non_lin('lkyrelu', 0.02),
            get_layer_norm('BN', hidden),
            nn.Linear(hidden, hidden),
        )
        self.fea_mlp_2 = nn.Sequential(
            nn.Linear(hidden,hidden),
            nn.Dropout(args['dropout']),
            get_non_lin('lkyrelu', 0.02),
            get_layer_norm('BN', hidden),
            nn.Linear(hidden, hidden),
        )

        self.r_mlp = nn.Sequential(
            nn.Linear(hidden*2+1,hidden*2),
            nn.Dropout(args['dropout']),
            get_non_lin('lkyrelu', 0.02),
            get_layer_norm('BN', hidden*2),
            nn.Linear(hidden*2, 1),
        )
        self.bern_1 = Bern_prop(K=10)


    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p, gain=1.)
            else:
                torch.nn.init.zeros_(p)

    def apply_edges1(self, edges):
        return {'cat_feat': torch.cat([edges.src['pro_h'], edges.dst['pro_h']], dim=1)}

    def forward(self, batch_1_graph, batch_2_graph):
        batch_2_graph.ndata['new_x_flex'] = batch_2_graph.ndata['new_x_flex'].to(torch.float)
        # batch_2_graph.ndata['new_x'] = batch_2_graph.ndata['x']       
        batch_1_graph.apply_edges(fn.u_sub_v('new_x_flex', 'new_x_flex', 'x_dis')) ## x_i - x_j
        batch_2_graph.apply_edges(fn.u_sub_v('new_x_flex', 'new_x_flex', 'x_dis'))
        edge_tmp_1 = batch_1_graph.edata['x_dis'] ** 2
        edge_tmp_2 = batch_2_graph.edata['x_dis'] ** 2
        edge_weight_1 = torch.sum(edge_tmp_1, dim=1, keepdim=True)   ## ||x_i - x_j||^2 : (N_res, 1)
        edge_weight_1 = torch.cat([torch.exp(-edge_weight_1 / sigma) for sigma in self.all_sigmas_dist], dim=-1)
        edge_weight_2 = torch.sum(edge_tmp_2, dim=1, keepdim=True)   ## ||x_i - x_j||^2 : (N_res, 1)
        edge_weight_2 = torch.cat([torch.exp(-edge_weight_2 / sigma) for sigma in self.all_sigmas_dist], dim=-1)


        # weight_lap_1 = F.relu(self.edge_mlp(torch.cat((batch_1_graph.edata['he'],edge_weight_1),dim=1)))
        # weight_lap_2 = F.relu(self.edge_mlp(torch.cat((batch_2_graph.edata['he'],edge_weight_2),dim=1)))
        weight_lap_1 = F.relu(self.edge_mlp(edge_weight_1))
        weight_lap_2 = F.relu(self.edge_mlp(edge_weight_2))
        
        edge_index_1 = torch.stack(batch_1_graph.edges())
        edge_index_2 = torch.stack(batch_2_graph.edges())
        
        
        batch_1_graph.apply_edges(self.apply_edges1)  ## i->j edge:  [h_i h_j]
        batch_2_graph.apply_edges(self.apply_edges1)

        batch_1_graph.edata['r_times_x_dis'] = self.r_mlp(torch.cat((batch_1_graph.edata['cat_feat'], \
                            weight_lap_1),dim=1)) * batch_1_graph.edata['x_dis']
        batch_2_graph.edata['r_times_x_dis'] = self.r_mlp(torch.cat((batch_2_graph.edata['cat_feat'], \
                            weight_lap_2),dim=1)) * batch_2_graph.edata['x_dis']
        
        batch_1_graph.update_all(fn.copy_edge('r_times_x_dis', 'm'), fn.mean('m', 'x_update'))
        batch_2_graph.update_all(fn.copy_edge('r_times_x_dis', 'm'), fn.mean('m', 'x_update'))

    

        batch_1_graph.ndata['new_x_flex'] = batch_1_graph.ndata['new_x_flex'] + batch_1_graph.ndata['x_update']
        batch_2_graph.ndata['new_x_flex'] = batch_2_graph.ndata['new_x_flex'] + batch_2_graph.ndata['x_update']
        
        batch_1_graph.ndata['pro_h'], TEMP_1 = self.bern_1(self.fea_mlp_1(batch_1_graph.ndata['pro_h']), edge_index_1.long(), weight_lap_1.T.squeeze(0))
        batch_2_graph.ndata['pro_h'], TEMP_2 = self.bern_1(self.fea_mlp_1(batch_2_graph.ndata['pro_h']), edge_index_2.long(), weight_lap_2.T.squeeze(0))
        
        # batch_1_graph.ndata['pro_h'], TEMP_1 = self.bern_2(self.fea_mlp_2(batch_1_graph.ndata['pro_h']), edge_index_1.long(), weight_lap_1.T.squeeze(0))
        # batch_2_graph.ndata['pro_h'], TEMP_2 = self.bern_2(self.fea_mlp_2(batch_2_graph.ndata['pro_h']), edge_index_2.long(), weight_lap_2.T.squeeze(0))

 
        # batch_1_graph.ndata['pro_h'], TEMP_1 = self.bern_1(self.fea_mlp_1(batch_1_graph.ndata['pro_h']), edge_index_1.long())
        # batch_2_graph.ndata['pro_h'], TEMP_2 = self.bern_1(self.fea_mlp_1(batch_2_graph.ndata['pro_h']), edge_index_2.long())
        
        # batch_1_graph.ndata['pro_h'] = self.fea_mlp_after(batch_1_graph.ndata['pro_h'])
        # batch_2_graph.ndata['pro_h'] = self.fea_mlp_after(batch_2_graph.ndata['pro_h'])
        # batch_1_graph.ndata['refine_x_1'] = self.bern_const(self.edge_mlp(torch.cat(batch_1_graph.ndata['edge_u_norm']


        return batch_1_graph.ndata['new_x_flex'], batch_1_graph.ndata['pro_h'], \
            batch_2_graph.ndata['new_x_flex'], batch_2_graph.ndata['pro_h'], TEMP_1, TEMP_2

    def __repr__(self):
        return "SEGCN Layer " + str(self.__dict__)

class CrossAttentionLayer(nn.Module):
    def __init__(self,args):
        # super().__init__()
        super(CrossAttentionLayer, self).__init__()
        self.h_dim = args['h_dim']
        self.h_dim_div = self.h_dim // 1
        self.num_heads = args['atten_head']
        assert self.h_dim_div % self.num_heads == 0
        self.head_dim = self.h_dim_div // self.num_heads
        self.merge = nn.Conv1d(self.h_dim_div, self.h_dim_div, kernel_size=1)
        self.proj = nn.ModuleList([nn.Conv1d(self.h_dim, self.h_dim_div, kernel_size=1) for _ in range(3)])
        dropout = args['dropout']

        self.mlp = nn.Sequential(
            nn.Linear(self.h_dim+self.h_dim_div, self.h_dim),
            nn.Dropout(dropout),
            nn.BatchNorm1d(self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.Dropout(dropout),
            nn.BatchNorm1d(self.h_dim)
        )
    def reset_parameters(self):
        if self.linear.weight is not None:
            init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            init.zeros_(self.linear.bias)

    def forward(self, src_h, dst_h, src_num_verts, dst_num_verts):
        h = dst_h
        src_h_list = torch.split(src_h, src_num_verts)
        dst_h_list = torch.split(dst_h, dst_num_verts)
        h_msg = []
        for idx in range(len(src_num_verts)):
            src_hh = src_h_list[idx].unsqueeze(0).transpose(1, 2)
            dst_hh = dst_h_list[idx].unsqueeze(0).transpose(1, 2)
            query, key, value = [ll(hh).view(1, self.head_dim, self.num_heads, -1) \
                for ll, hh in zip(self.proj, (dst_hh, src_hh, src_hh))]
            dim = query.shape[1]
            scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / (dim ** 0.5)
            attn = F.softmax(scores, dim=-1)
            h_dst = torch.einsum('bhnm,bdhm->bdhn', attn, value) 
            h_dst = self.merge(h_dst.contiguous().view(1, self.h_dim_div, -1))
            h_msg.append(h_dst.squeeze(0).transpose(0, 1))
        h_msg = torch.cat(h_msg, dim=0)

        # skip connection
        h_out = h + self.mlp(torch.cat((h, h_msg), dim=-1))

        return h_out


# =================================================================================================================
class SEGCN(nn.Module):

    def __init__(self, args, n_lays, fine_tune, log=None):

        super(SEGCN, self).__init__()

        self.debug = args['debug']
        self.log=log

        self.device = args['device']
        self.graph_nodes = args['graph_nodes']

        self.rot_model = args['rot_model']

        self.noise_decay_rate = args['noise_decay_rate']
        self.noise_initial = args['noise_initial']

        # 21 types of amino-acid types
        self.residue_emb_layer = nn.Embedding(num_embeddings=21, embedding_dim=args['residue_emb_dim'])

        assert self.graph_nodes == 'residues'

        self.segcn_layers = nn.ModuleList()
        
     
        self.segcn_layers.append(SEGCN_Layer(args))

       
        self.c_a_layer = CrossAttentionLayer(args)

        self.n_layer =  args['SEGCN_layer']
        if self.n_layer > 1:
            interm_lay = SEGCN_Layer(args)
            for layer_idx in range(1, self.n_layer):
                self.segcn_layers.append(interm_lay)
        



        assert args['rot_model'] == 'kb_att'

       
        self.fea_norm_mlp = nn.Sequential(
            nn.Linear(64,args['h_dim']),
            nn.Dropout(args['dropout']),
            get_non_lin('lkyrelu', 0.02),
            get_layer_norm('BN', args['h_dim']),
            nn.Linear(args['h_dim'], args['h_dim']),
            get_non_lin('lkyrelu', 0.02),
            get_layer_norm('BN', args['h_dim']),
        )

        
        self.interface_clsf = nn.Sequential(
            nn.Linear(args['h_dim'], args['h_dim']),
            nn.Dropout(args['dp']),
            get_non_lin(args['nonlin'], args['leakyrelu_neg_slope']),
            get_layer_norm('BN', args['h_dim']),
            
            # nn.Sigmoid(),
        )
        # self.reset_parameters()
        self.clsf = nn.Sequential(
            nn.Linear(args['h_dim'], args['h_dim']),
            nn.Dropout(args['dp']),
            # get_non_lin(args['nonlin'], args['leakyrelu_neg_slope']),
            get_non_lin('lkyrelu', 0.02),
            get_layer_norm('BN', args['h_dim']),
            nn.Linear(args['h_dim'], 2),
            nn.Sigmoid(),
            # nn.Softmax()
        )

        

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p, gain=1.)
            else:
                torch.nn.init.zeros_(p)


    def forward(self, batch_1_graph, batch_2_graph, train_tuple, epoch):



        ## Embed residue types with a lookup table.
        h_feats_1 = self.residue_emb_layer(
            batch_1_graph.ndata['res_feat'].view(-1).long())  # (N_res, emb_dim)

        h_feats_2 = self.residue_emb_layer(
            batch_2_graph.ndata['res_feat'].view(-1).long())  # (N_res, emb_dim)

        # h_feats_1 = torch.cat([h_feats_1, torch.log(batch_1_graph.ndata['mu_r_norm'])], dim=1)
        # h_feats_2 = torch.cat([h_feats_2, torch.log(batch_2_graph.ndata['mu_r_norm'])], dim=1)


        batch_1_graph.ndata['pro_h'] = self.fea_norm_mlp(h_feats_1)
        # batch_1_graph.ndata['x_before_dy'] = batch_1_graph.ndata['new_x_flex']
        batch_2_graph.ndata['pro_h'] = self.fea_norm_mlp(h_feats_2)

        for i, layer in enumerate(self.segcn_layers):
            coors_ligand, h_feats_ligand, coors_receptor, h_feats_receptor, \
                TEMP_1, TEMP_2 = layer(batch_1_graph,batch_2_graph)

        batch_1_graph.ndata['x_segcn_out'] = coors_ligand
        batch_2_graph.ndata['x_segcn_out'] = coors_receptor
        batch_1_graph.ndata['hv_segcn_out'] = h_feats_ligand
        batch_2_graph.ndata['hv_segcn_out'] = h_feats_receptor
        pre_interface_batch = []
        list_graph_1 = dgl.unbatch(batch_1_graph)
        list_graph_2 = dgl.unbatch(batch_2_graph)
        for ii in range(len(train_tuple)):
            train_tuple_sg = train_tuple[ii].long()
            h_1 = list_graph_1[ii].ndata['hv_segcn_out']
            h_2 = list_graph_2[ii].ndata['hv_segcn_out']
            # pre_interface = torch.mul(self.interface_clsf(h_1[train_tuple_sg[0]]),self.interface_clsf(h_2[train_tuple_sg[1]]))
            # pre_interface = torch.cat((self.interface_clsf(h_1[train_tuple_sg[0]]),self.interface_clsf(h_2[train_tuple_sg[1]])),dim=1)
            
            h_2_ca = self.c_a_layer(h_1,h_2,[h_1.size(0)],[h_2.size(0)])
            h_1_ca = self.c_a_layer(h_2,h_1,[h_2.size(0)],[h_1.size(0)])
            
            pre_interface = torch.mul(h_1_ca[train_tuple_sg[0]],h_2_ca[train_tuple_sg[1]])
            pre_interface = self.clsf(pre_interface)
            pre_interface_batch.append(pre_interface)        

        return [TEMP_1, TEMP_2, pre_interface_batch, batch_1_graph, batch_2_graph]


    def __repr__(self):
        return "SEGCN " + str(self.__dict__)

# =================================================================================================================
class REFINE_MODULE(nn.Module):

    def __init__(self,args):

        super(REFINE_MODULE, self).__init__()

        self.r_mlp = nn.Sequential(
            nn.Linear(args['h_dim']*2,args['h_dim']*2),
            nn.Dropout(args['dp']),
            get_non_lin('lkyrelu', 0.02),
            get_layer_norm('BN', args['h_dim']*2),
            nn.Linear(args['h_dim']*2, 1),
        )
    
    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p, gain=1.)
            else:
                torch.nn.init.zeros_(p)

    def apply_edges1(self, edges):
        return {'cat_feat': torch.cat([edges.src['h'], edges.dst['h']], dim=1)}

    def forward(self, graph_1_dock, graph_2_dock):
        
        batch_1_2_dock = dgl.batch([graph_1_dock, graph_2_dock])

        new_KNN_graph = dgl.knn_graph(batch_1_2_dock.ndata['new_x_flex'], k=10)
        new_KNN_graph.ndata['x'] = batch_1_2_dock.ndata['new_x_flex']
        new_KNN_graph.ndata['h'] = batch_1_2_dock.ndata['hv_segcn_out']

        new_KNN_graph.apply_edges(fn.u_sub_v('x', 'x', 'x_dis')) ## x_i - x_j
        
        new_KNN_graph.apply_edges(self.apply_edges1)  ## i->j edge:  [h_i h_j]

        new_KNN_graph.edata['r_times_x_dis'] = self.r_mlp(new_KNN_graph.edata['cat_feat']) * new_KNN_graph.edata['x_dis']
        
        new_KNN_graph.update_all(fn.copy_edge('r_times_x_dis', 'm'), fn.mean('m', 'x_update'))
    
        new_KNN_graph.ndata['x'] = new_KNN_graph.ndata['x'] + new_KNN_graph.ndata['x_update']

        return dgl.unbatch(batch_1_2_dock)[0].ndata['new_x_flex'], dgl.unbatch(batch_1_2_dock)[1].ndata['new_x_flex']



class FLEXDOCK_MODEL(nn.Module):

    def __init__(self, args, log=None):

        super(FLEXDOCK_MODEL, self).__init__()

        self.debug = args['debug']
        self.log=log
        self.args = args
        self.device = args['device']

        self.segcn_original = SEGCN(args, n_lays=args['SEGCN_layer'], fine_tune=False, log=log)
        if args['fine_tune']:
            self.segcn_fine_tune = SEGCN(args, n_lays=2, fine_tune=True, log=log)
            self.list_segcns = [('original', self.segcn_original), ('finetune', self.segcn_fine_tune)]
        else:
            self.list_segcns = [('finetune', self.segcn_original)] ## just original
        
        self.refine = REFINE_MODULE(args)

        # self.reset_parameters()
        kab_dim = args['kab_dim']
        

        self.wei_kab = nn.Sequential(
            nn.Linear(args['h_dim']*2, kab_dim),
            nn.Dropout(args['dropout']),
            get_non_lin(args['nonlin'], args['leakyrelu_neg_slope']),
            # get_layer_norm('BN', kab_dim),
            nn.Linear(kab_dim, 1),
            nn.Sigmoid()
        )

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p, gain=1.)
            else:
                torch.nn.init.zeros_(p)



    ####### FORWARD for Rigid_Body_Docking_Net
    def forward(self, batch_ligand_graph, batch_receptor_graph, train_tuple, train_label_tuple, epoch):
        last_outputs = None
        all_ligand_coors_deform_list = []
        x_1_final_list = []
        x_2_final_list = []
        # obtain interface:
        # with torch.no_grad():
        for stage, segcn in self.list_segcns:
            outputs = segcn(batch_ligand_graph, batch_receptor_graph, train_tuple, epoch)
            
        # perform weighted kabsch

        list_graph_1 = dgl.unbatch(outputs[3])
        list_graph_2 = dgl.unbatch(outputs[4])
        for jj in range(len(list_graph_1)):
            h_1 = list_graph_1[jj].ndata['hv_segcn_out']
            h_2 = list_graph_2[jj].ndata['hv_segcn_out']
            z_1 = list_graph_1[jj].ndata['x_segcn_out']
            z_2 = list_graph_2[jj].ndata['x_segcn_out']
            x_1_dy = list_graph_1[jj].ndata['new_x_flex']
            x_2_dy = list_graph_2[jj].ndata['new_x_flex']
            # percen_median = 0.2
            # pre_int = torch.where(torch.argmax(outputs[2][jj],1)==1)[0]
            # if epoch>50:
            percen_median = np.percentile(np.array(outputs[2][jj][:,1].detach().cpu()),60)
            
            pre_int = torch.where(outputs[2][jj][:,1]>percen_median)[0]
            # ind_b = train_tuple[jj].shape[1]*(train_label_tuple[0].mean())
            pre_int_ind = train_tuple[jj].T[pre_int].long()
            # pre_int_ind = train_tuple[jj].T[:ind_b.long(),:].long()
            # random.shuffle(pre_int_ind[:,1])
            h_1_2 = torch.cat((h_1[pre_int_ind[:,0]],h_2[pre_int_ind[:,1]]),dim = 1)
            z_1_int = x_1_dy[pre_int_ind[:,0]]
            z_2_int = x_2_dy[pre_int_ind[:,1]]
            device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
            # w_vector = torch.ones((h_1_2.size(0),1)).to(device)
            # if epoch>50:
            # w_vector = self.wei_kab(h_1_2)
            # w_matrix = torch.diag_embed(w_vector.T).squeeze(0)
            R,t = rigid_transform_Kabsch_3D_model(z_1_int.T, z_2_int.T, self.device)
            x_1_rigid = R @ list_graph_1[jj].ndata['new_x_flex'].T + t
            x_1_rigid = x_1_rigid.T
            x_2_rigid = list_graph_2[jj].ndata['new_x_flex']
            if self.args['stage_2'] == True:
                list_graph_1[jj].ndata['new_x_flex'], list_graph_2[jj].ndata['new_x_flex']  = x_1_rigid, x_2_rigid
                x_1_final, x_2_final = self.refine(list_graph_1[jj], list_graph_2[jj])
            else:
                x_1_final, x_2_final = x_1_rigid, x_2_rigid
                 
            x_1_final_list.append(x_1_final)
            x_2_final_list.append(x_2_final)

        return outputs[0], outputs[1], outputs[2], x_1_final_list, x_2_final_list


    def __repr__(self):
        return "FLEXDOCK_MODEL " + str(self.__dict__)


def rigid_transform_Kabsch_3D_model(A, B,device):
    assert A.shape[1] == B.shape[1]
    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")
    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")


    # find mean column wise: 3 x 1
    centroid_A = A.mean(dim=1, keepdims=True)
    centroid_B = B.mean(dim=1, keepdims=True)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    HA = Am @ Bm.T

    # find rotation
    # assert not torch.isnan(HA).any()
    U, S, Vt = torch.linalg.svd(HA)

    num_it = 0
    while torch.min(S) < 1e-3 or torch.min(torch.abs((S**2).view(1,3) - (S**2).view(3,1) + torch.eye(3).to(device))) < 1e-2:

        HA = HA + torch.rand(3,3).to(device) * torch.eye(3).to(device)
        U, S, Vt = torch.linalg.svd(HA)
        num_it += 1

        if num_it > 10:
            sys.exit(1)

    corr_mat = torch.diag(torch.Tensor([1,1,torch.sign(torch.det(HA))])).to(device)
    T_align = Vt.T @ U.T

    b_align = centroid_B - T_align @ centroid_A  # (1,3)
    return T_align, b_align

def rigid_transform_Kabsch_3D_model_copy(A, B, w_matrix,device):
    assert A.shape[1] == B.shape[1]
    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")
    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")


    # find mean column wise: 3 x 1
    centroid_A = A.mean(dim=1, keepdims=True)
    centroid_B = B.mean(dim=1, keepdims=True)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    HA = Am @  w_matrix  @ Bm.T

    # find rotation
    assert not torch.isnan(HA).any()
    U, S, Vt = torch.linalg.svd(HA)

    num_it = 0
    while torch.min(S) < 1e-3 or torch.min(torch.abs((S**2).view(1,3) - (S**2).view(3,1) + torch.eye(3).to(device))) < 1e-2:

        HA = HA + torch.rand(3,3).to(device) * torch.eye(3).to(device)
        U, S, Vt = torch.linalg.svd(HA)
        num_it += 1

        if num_it > 10:
            sys.exit(1)

    corr_mat = torch.diag(torch.Tensor([1,1,torch.sign(torch.det(HA))])).to(device)
    T_align = (U @  corr_mat) @ Vt

    b_align = centroid_B - T_align @ centroid_A  # (1,3)
    return T_align, b_align



