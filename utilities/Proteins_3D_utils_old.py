"""
 Copyright 2020 - by Jerome Tubiana (jertubiana@gmail.com)
     All rights reserved

     Permission is granted for anyone to copy, use, or modify this
     software for any uncommercial purposes, provided this copyright
     notice is retained, and note is made of any changes that have
     been made. This software is distributed without any warranty,
     express or implied. In no event shall the author or contributors be
     liable for any damage arising out of the use of this software.

     The publication of research using this software, modified or not, must include
     appropriate citations to:
"""

import sys
sys.path.append('../source/')
from Bio import pairwise2
import os
import numpy as np
import Proteins_utils
import utilities

d = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
     'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
     'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
     'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

d_inv = dict([(item,key) for key,item in d.items()])


def write_PDB(fil,subset,sequence_ref, pos_Calpha):
    N = len(sequence_ref)
    if not type(sequence_ref) == str:
        sequence_ref = Proteins_utils.num2seq(sequence_ref)

    with open(fil,'w') as f:
        for k,i in enumerate(subset):
            aa = d_inv[sequence_ref[i]]
            pos = pos_Calpha[i]
            line = 'ATOM'
            line += format(k+1,' 7d')
            line += '  CA  %s A'%aa
            line += format(k+1,' 4d')
            line+= format(pos[0], ' 12.3f')
            line += format(pos[1], ' 8.3f')
            line += format(pos[2], ' 8.3f')
            line += '  1.00 99.99           C  \n'
            f.write(line)
    return 'done'


def color_PDB(fil,fil_out,chain,has_repeat=False):
    header = []
    all_lines = []
    all_num = []
    if has_repeat:
        saw_ter = False
    with open(fil,'r') as f:
        for line in f:
            if line[:4] == 'ATOM':
                line_num = int(line[23:26])
                line_chain = line[21]
                if has_repeat:
                    if saw_ter:
                        chain_id = chain(-1)
                    else:
                        chain_id = chain(line_num)
                else:
                    chain_id = chain(line_num)
                line_final = line[:21] + chain_id + line[22:]
                all_lines.append(line_final)
                all_num.append(line_num)
            elif line[:3] == 'TER':
                if has_repeat:
                    if saw_ter:
                        break
                    else:
                        saw_ter = True

            elif line[:4] == 'ANIS':
                continue
            else:
                header.append(line)

    with open(fil_out,'wb') as f:
        for line in header:
            f.write(line)
        for line in all_lines:
            f.write(line)
    return 'done'





def load_PDB(fil,chain=None,maxiline=int(1e6),return_distance=True,has_repeat=False):
    f = open(fil,'r')
    pos = []
    pos_Calpha = []
    aa = []
    num = []
    atom_index = []

    current_num = 0

    for k in range(maxiline):
        try:
            line = f.readline()

            if chain is not None:
                use_it = (line[:4] == 'ATOM') & (line[21] == chain)
            else:
                use_it = (line[:4] == 'ATOM')


            if use_it:
                line_num = int(line[23:26])
                if line_num != current_num:
                    current_num = line_num
                    num.append(current_num)
                    aa.append(line[17:20])
                    pos.append([])

                atom = line[13:15].replace(' ','')
                if atom != 'H':
                    x = float(line[28:38])
                    y = float(line[38:46])
                    z = float(line[46:55])
                    pos[-1].append([x,y,z])
                    if atom == 'CA':
                        pos_Calpha.append([x,y,z])
                        atom_index.append(int(line[4:11]))
        except:
            break


    maxi = len(num)

    if return_distance:
        distance = np.zeros([maxi,maxi])
        for i in range(maxi):
            for j in range(maxi):
                distance[i,j]=np.sqrt( ((np.array(pos[i])[np.newaxis,:,:] - np.array(pos[j])[:,np.newaxis,:])**2).sum(-1) ).min()

    sequence_ref = ''.join([d[a] for a in aa])
    pos_Calpha = np.array(pos_Calpha)
    if return_distance:
        return sequence_ref, pos_Calpha, distance, atom_index
    else:
        return sequence_ref, pos_Calpha, atom_index



def learn_mapping_to_alignment(alignment, sequence,
                  hmm_path = '/Users/jerometubiana/Desktop/PhD/hmmer-3.2.1/',
                  n_iter = 3,
                 verbose=1):

    if not type(alignment) == str: # data alignment.
        name_alignment = 'tmp.fasta'
        sequences_alignment = alignment
        Proteins_utils.write_FASTA('tmp.fasta',alignment)
    else:
        name_alignment = alignment
        sequences_alignment = Proteins_utils.load_FASTA(alignment,drop_duplicates=False)
    sequences_alignment_original = sequences_alignment.copy()
    consensus_sequence = np.argmax(utilities.average(sequences_alignment_original,c=21)[:,:-1],axis=1)[np.newaxis,:]
    if type(sequence) == str:
        sequence_num = Proteins_utils.seq2num(sequence)
    else:
        sequence_num = sequence
        if sequence_num.ndim==1:
            sequence_num = sequence_num[np.newaxis]

    Proteins_utils.write_FASTA('tmp_target.fasta',sequence_num)

    for iteration in range(1,n_iter+1):
        hmm_alignment = 'tmp.hmm'
        if iteration>1:
            cmd = hmm_path+'src/hmmbuild --symfrac 0 --wnone %s %s'%(hmm_alignment,name_alignment)
        else:
            cmd = hmm_path+'src/hmmbuild --symfrac 0 %s %s'%(hmm_alignment,name_alignment)
        os.system(cmd)
        cmd = hmm_path+'src/hmmalign -o tmp_aligned.txt %s %s'%(hmm_alignment,'tmp_target.fasta')
        os.system(cmd)
        cmd = hmm_path+'easel/miniapps/esl-reformat --informat stockholm afa tmp_aligned.txt > tmp_aligned.fasta'
        os.system(cmd)
        sequence_aligned = ''.join(open('tmp_aligned.fasta','r').read().split('\n')[1:])
        if verbose:
            print('Iteration %s: %s,'%(iteration,sequence_aligned) )
        mapping_alignment_to_struct = []
        sequence_ref_aligned = []
        index_sequence = 0
        index_alignment = 0
        for k,s in enumerate(sequence_aligned):
            if s == '-':
                mapping_alignment_to_struct.append(-1)
                index_alignment +=1
                sequence_ref_aligned.append('-')
            elif s == s.upper():
                mapping_alignment_to_struct.append(index_sequence)
                index_sequence +=1
                index_alignment +=1
                sequence_ref_aligned.append(s)
            elif s==s.lower():
                index_sequence +=1

        mapping_alignment_to_struct = np.array(mapping_alignment_to_struct,dtype='int')
        print(len(sequence_ref_aligned))
        sequence_ref_aligned = Proteins_utils.seq2num(''.join(sequence_ref_aligned))
        if verbose:
            fraction_of_sites = (mapping_alignment_to_struct!=-1).mean()
            print('Iteration %s, fraction of sites mapped on the structure: %.2f'%(iteration,fraction_of_sites))

        top_closest = np.minimum(50, sequences_alignment_original.shape[0]//5)
        closest = np.argsort( (sequences_alignment_original == sequence_ref_aligned ).mean(1) )[::-1][:top_closest]
        name_alignment = 'tmp.fasta'
        reduced_alignment = np.concatenate(
            (np.repeat( sequences_alignment_original[closest], 10,axis=0 ),
            consensus_sequence) , axis=0) # Need to add the consensus sequence. Otherwise, hmmalign can remove a column if it has only gaps in the reduced alignment. compensate by increasing the weights of the other sequences and removing the reweighting.
        Proteins_utils.write_FASTA('tmp.fasta',reduced_alignment )
    os.system('rm tmp_target.fasta tmp_aligned.txt tmp_aligned.fasta tmp.hmm tmp.fasta')
    return mapping_alignment_to_struct,sequence_ref_aligned




def learn_mapping_pairwise(sequence1, sequence2):
    if type(sequence1) != str:
        sequence1 = Proteins_utils.num2seq(sequence1)
    if type(sequence2) != str:
        sequence2 = Proteins_utils.num2seq(sequence2)

    alignments = pairwise2.align.globalxx(sequence1,sequence2 )

    al1 = alignments[0][0]
    al2 = alignments[0][1]


    mapping1 = -np.ones(len(sequence1),dtype='int')
    mapping1to2 = -np.ones(len(sequence1),dtype='int')
    mapping2 = -np.ones(len(sequence2),dtype='int')
    mapping2to1 = -np.ones(len(sequence2),dtype='int')

    current = 0
    for k,letter in enumerate(sequence1):
        found = False
        while not found:
            if current >= len(al1):
                found = True
                break
            if letter == al1[current]:
                found = True
                mapping1[k] = current
            current+=1


    current = 0
    for k,letter in enumerate(sequence2):
        found = False
        while not found:
            if current >= len(al2):
                found = True
                break
            if letter == al2[current]:
                found = True
                mapping2[k] = current
            current+=1

    for k in range(len(mapping1to2)):
        tmp = mapping1[k]
        try:
            mapping1to2[k] = np.nonzero(mapping2 == tmp)[0]
        except:
            mapping1to2[k] = -1

    for k in range(len(mapping2to1)):
        tmp = mapping2[k]
        try:
            mapping2to1[k] = np.nonzero(mapping1 == tmp)[0]
        except:
            mapping2to1[k] = -1
    return mapping1to2, mapping2to1


def visualize_sectors(sectors, pdb_file, output_folder, name_VMD,
                      sector_names = None,
                      mapping = None,
                      sequence_pdb_aligned = None,
                      alignment = None,
                     vmd_path = '"/Applications/VMD 1.9.4.app/Contents/vmd/vmd_MACOSXX86"',
                      tachyon_path = '"/Applications/VMD 1.9.4.app/Contents/vmd/tachyon_MACOSXX86"',
                      hmm_path = '/Users/jerometubiana/Desktop/PhD/hmmer-3.2.1/',
                    color_mode = 'Chain' , sector_colors = None,
                      with_numbers=True,with_numbers_every = 2,offset_labels=0.05,label_size=1.0,
                      view_point=None, render=False, pixel_size = 1000,
                      render_method = 'OpenGL',fordimeronly=None,chain=None):

    output_folder = os.path.abspath(output_folder)
    output_folder_safe_spaces = output_folder.replace(" ", "\ ")

    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    os.system('scp %s %s/'%(pdb_file,output_folder_safe_spaces) )
    pdb_file = pdb_file.split('/')[-1]
    print('loading PDB file....')
    sequence_pdb, pos_Calpha,atom_index = load_PDB(output_folder+'/'+pdb_file,has_repeat=False,return_distance=False,chain=chain)

    if (mapping is None):
        if alignment is not None:
            print('Learning the mapping between MSA columns and PDB sequence index from the alignment...')
            mapping,sequence_pdb_aligned = learn_mapping_to_alignment(alignment, sequence_pdb,hmm_path=hmm_path)
        elif sequence_pdb_aligned is not None:
            print('Learning the mapping between MSA columns and PDB sequence index from the aligned PDB sequence...')
            _ , mapping = learn_mapping_pairwise(sequence_pdb,sequence_pdb_aligned)
        else:
            print('Must provide either i) The PDB sequence as it appears in the alignment (sequence_pdb_aligned) ii) The entire alignment (alignment) iii) The mapping between the alignment columns and the PDB sequence (mapping)')
            return

        print('Mapping learnt...')
    else:
        print('Mapping provided...')

    if sector_names is None:
        if len(sectors) ==1:
            sector_names = ['']
        else:
            sector_names = ['_sector_%s'%k for k in range(1,len(sectors)+1)]
    else:
        sector_names = ['_'+sector_name for sector_name in sector_names]

    if sector_colors is None:
        sector_colors = ['7' for _ in sectors]



    lines_VMD = ['mol new %s/%s\n'%(output_folder_safe_spaces,pdb_file)]
    lines_VMD.append('mol delrep 0 top\n')
    lines_VMD.append('mol representation NewCartoon\n')
    lines_VMD.append('mol color %s\n'%color_mode)
    lines_VMD.append('color Chain {A} blue\n')
    # lines_VMD.append('color Chain {B} cyan\n')
    lines_VMD.append('color Chain {B} blue\n')
    lines_VMD.append('color Chain {C} red\n')
    lines_VMD.append('color Chain {D} gray\n')
    # lines_VMD.append('color Chain {E} yellow\n')
    lines_VMD.append('mol addrep 0\n')
    lines_VMD.append('color Display {Background} white\n')
    lines_VMD.append('color Labels {Atoms} black\n')
    lines_VMD.append('label textsize %.1f\n'%label_size)
    lines_VMD.append('label textthickness 2.5\n')


    count_labels = 0
    count_sectors = 0
    if with_numbers:
        already_done = []
        lines_numbers_VMD = []

    if render_method is None:
        if with_numbers:
            render_method = 'OpenGL'
        else:
            render_method = 'Tachyon'
    for sector,sector_name in zip(sectors,sector_names):
        name = pdb_file.split('_')[0].split('.')[0] + '_' + name_VMD + sector_name +'.pdb'
        subset = mapping[sector]
        sector_ = sector[subset>=0]
        subset = subset[subset>=0]
        print(sector_name,subset)
        write_PDB(output_folder+'/'+name, subset, sequence_pdb, pos_Calpha)
        lines_VMD.append('mol new %s/%s\n'%(output_folder_safe_spaces,name))

        if with_numbers:
            for k,i_align in enumerate(sector_):
                if i_align not in already_done:
                    if with_numbers_every>0:
                        already_done += range(i_align - with_numbers_every, i_align + with_numbers_every +1)
                    else:
                        already_done.append(i_align)
                    number_every = 0
                    lines_numbers_VMD.append('label add Atoms %s/%s\n'%(count_sectors+1,k))
                    lines_numbers_VMD.append('label textoffset Atoms %s {%.2f %.2f}\n'%(count_labels,offset_labels,offset_labels))
                    if fordimeronly is not None:
                        value_label = (i_align%fordimeronly+1)
                    else:
                        value_label = (i_align+1)
                    lines_numbers_VMD.append('label textformat Atoms %s %s\n'%(count_labels,value_label))
                    count_labels+=1

        count_sectors +=1

    for k in range(1,len(sectors)+1):
        lines_VMD.append('mol delrep %s top\n'%k)
    lines_VMD.append('mol representation VDW\n')
    for k in range(len(sectors)):
        lines_VMD.append('mol Color ColorID %s\n'%sector_colors[k])
        lines_VMD.append('mol addrep %s\n'%(k+1))
    for k in range(len(sectors)+1):
        lines_VMD.append('mol on %s\n'%k)

    lines_VMD.append('axes location off\n')

    if view_point is not None:
        for k in range(len(sectors)+1):

            lines_VMD.append('molinfo %s set {center_matrix} {%s}\n'%(k,view_point['center_matrix']) )
            lines_VMD.append('molinfo %s set {scale_matrix} {%s}\n'%(k,view_point['scale_matrix']) )
            lines_VMD.append('molinfo %s set {rotate_matrix} {%s}\n'%(k,view_point['rotate_matrix']) )
            lines_VMD.append('molinfo %s set {global_matrix} {%s}\n'%(k,view_point['global_matrix']) )

    else:
        lines_VMD.append('mol new %s/%s\n'%(output_folder_safe_spaces,pdb_file))
        lines_VMD.append('mol off %s\n'%(len(sectors)+1) )


    if with_numbers:
        lines_VMD += lines_numbers_VMD

    if render:
        tachyon_file = output_folder_safe_spaces + '/'+ pdb_file.split('_')[0].split('.')[0] + '_' + name_VMD + '.tachyon'
        tga_file = output_folder_safe_spaces + '/'+ pdb_file.split('_')[0].split('.')[0] + '_' + name_VMD + '.tga'
        if render_method == 'Tachyon':
            lines_VMD.append('render Tachyon %s'%tachyon_file)
        else:
            lines_VMD.append('render snapshot %s'%tga_file)

    vmd_file = output_folder + '/'+ pdb_file.split('_')[0].split('.')[0] + '_' + name_VMD + '.vmd'
    vmd_file_safe_spaces = output_folder_safe_spaces + '/'+ pdb_file.split('_')[0].split('.')[0] + '_' + name_VMD + '.vmd'
    with open(vmd_file,'w') as f:
        for line in lines_VMD:
            f.write(line)
    if render:
        if type(pixel_size) != list:
            pixel_size = [pixel_size,pixel_size]
        print('Start rendering...')
        command = '%s -size %s %s -e %s'%(vmd_path,pixel_size[0],pixel_size[1],vmd_file_safe_spaces)
        print(command)
        os.system(command)
        if render_method == 'Tachyon':
            command = '%s -aasamples 12 %s -format TARGA -o %s'%(tachyon_path,tachyon_file,tga_file)
            os.system(command)
        print('Rendering done...')
    print('play %s'%vmd_file)
    print('All done')
    return mapping
