import struct
import os
if __name__ == '__main__':
    filepath='/apdcephfs/share_1364275/kaithgao/equidock_public/cache/db5_residues_maxneighbor_10_cutoff_30.0_pocketCut_8.0/cv_0/ligand_graph_test.bin'
    binfile = open(filepath, 'rb') #打开二进制文件
    size = os.path.getsize(filepath) #获得文件大小
    for i in range(size):
        data = binfile.read(1) #每次输出一个字节
        print(data)
    binfile.close()