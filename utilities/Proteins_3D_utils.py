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
from Bio.PDB import PDBParser,PDBList,Selection
import os
import numpy as np
import Proteins_utils
import utilities
import copy


chimera_path = '/Applications/Chimera.app/Contents/MacOS/chimera' # Path to chimera command line executable.
hmmer_path = '/Users/jerometubiana/Desktop/PhD/hmmer-3.2.1/' # Path to executables should be, for instance: hmmer_path+'src/hmmbuild'
structures_folder = '/Volumes/Carte_SD/PDB_files/' # Where PDB files are stored locally.
residue_dictionary = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
                      'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
                      'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
                      'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M', 'MSE': 'M'}




def is_residue(residue):
    try:
        return (residue.get_id()[0] == ' ') & (residue.resname in residue_dictionary.keys())
    except:
        return False



def learn_mapping_to_alignment(alignment, sequence,
                  hmmer_path = hmmer_path,
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
            cmd = hmmer_path+'src/hmmbuild --symfrac 0 --wnone %s %s'%(hmm_alignment,name_alignment)
        else:
            cmd = hmmer_path+'src/hmmbuild --symfrac 0 %s %s'%(hmm_alignment,name_alignment)
        os.system(cmd)
        cmd = hmmer_path+'src/hmmalign -o tmp_aligned.txt %s %s'%(hmm_alignment,'tmp_target.fasta')
        os.system(cmd)
        cmd = hmmer_path+'easel/miniapps/esl-reformat --informat stockholm afa tmp_aligned.txt > tmp_aligned.fasta'
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



def visualize_sectors(sectors,pdb,output_folder,
                      chain=None, sector_names = None,
                      mapping=None,alignment = None,
                      sequence_pdb_aligned = None,
                      sector_colors = None,
                      chain_colors = None,
                       first_model_only =False,
                      chimera_path = chimera_path,
                      hmmer_path = hmmer_path,
                      simultaneous=True, save=True,npixels = 1000,
                      with_numbers=True,with_numbers_every = 2,
                      pdb_numbers = False,exit=False,
                     turn=None,show_sidechains=True):

    '''
    - To adjust label fontsize: open chimera --> Favorites --> Preferences --> Labels --> Change default label size --> Save.
    - To adjust viewpoint:
    1. Run the script first with exit=False.
    2. Open command line (Favorites --> Command line).
    3. Play with the reset,turn,focus commands (https://www.cgl.ucsf.edu/chimera/current/docs/UsersGuide/framecommand.html) until reaching a satisfying result:
    reset
    turn 1,0,0 45
    reset
    turn 1,0,0 90
    reset
    turn 1,0,0 90 center 0,0,0; focus;
    ...

    Ex, PDB 2kho, final command: turn 0,1,0 270; focus; turn 1,0,0 180; turn 0,0,1 20; focus;


    4. Pass the command as a string optional argument to visualize_sector: visualize_sector(...,turn = 'turn 0,1,0 270; focus; turn 1,0,0 180; turn 0,0,1 20; focus;')

    '''

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    try:
        x = sectors[0][0] # list of sectors.
    except:
        sectors = [sectors]

    sectors = [np.array(sector) for sector in sectors]

    if sector_names is None:
        sector_names = ['%s'%(k+1) for k in range(len(sectors))]


    if len(pdb) == 4:
        print('%s is a PDB id'%pdb)
        pdb_file = structures_folder + 'pdb' + pdb.lower() + '.ent'
        if not os.path.exists(pdb_file):
            print('%s not found, downloading it from PDB...'%pdb)
            pdbl = PDBList()
            pdb_file = pdbl.retrieve_pdb_file(pdb,pdir=structures_folder,file_format='pdb')
    else:
        pdb_file = pdb
    assert os.path.exists(pdb_file),'pdb file not found'

    parser = PDBParser()
    structure = parser.get_structure('1234',  pdb_file)
    nmodels = len(Selection.unfold_entities(structure,'M'))
    chain_objs = Selection.unfold_entities(structure[0],'C') # Take all chains of the first model.
    nchains = len(chain_objs)
    if chain is not None:
        # Discard all non-relevant chains.
        if not type(chain) == list:
            chain = [chain]

        i = 0
        while i < nchains:
            chain_id = chain_objs[i].get_full_id()[2]
            if not chain_id in chain:
                del chain_objs[i]
                nchains -= 1
            else:
                i+=1
    list_chains = [chain_obj.get_full_id()[2]  for chain_obj in chain_objs]
    print('%s has %s chains'%(pdb_file,  nchains) )
    print(list_chains)




    residues = Selection.unfold_entities(chain_objs,'R')
    sequence_pdb = ''
    index_pdb = []
    for residue in residues:
        if is_residue(residue):
            sequence_pdb += residue_dictionary[residue.resname]
            index_pdb.append( (residue.get_full_id()[2],residue.get_id()[1]) )


    if (mapping is None):
        if alignment is not None:
            print('Learning the mapping between MSA columns and PDB sequence index from the alignment...')
            mapping,sequence_pdb_aligned = learn_mapping_to_alignment(alignment, sequence_pdb,hmmer_path=hmmer_path)
        elif sequence_pdb_aligned is not None:
            print('Learning the mapping between MSA columns and PDB sequence index from the aligned PDB sequence...')
            _ , mapping = learn_mapping_pairwise(sequence_pdb,sequence_pdb_aligned)
        else:
            print('Must provide either i) The PDB sequence as it appears in the alignment (sequence_pdb_aligned) ii) The entire alignment (alignment) iii) The mapping between the alignment columns and the PDB sequence (mapping)')
            return

        print('Mapping learnt...')
    else:
        print('Mapping provided...')



    commands_chimera = []
    commands_chimera.append('open %s'%pdb_file)

    if nmodels>1:
        model = 1
    else:
        model = 0

    if chain_colors is None:
        chain_colors = [
            'red',
            'yellow',
            'blue',
            'orange']

    if sector_colors is None:
        sector_colors = [
            'green',
            'black',
            'blue',
            'orange'
        ]

    if first_model_only & (nmodels>1):
        for model_ in range(2,nmodels+1):
            commands_chimera.append('close #0.%s'%model_)

    if turn is not None:
        commands_chimera.append('reset')
        commands_chimera += turn.split(';')

    commands_chimera.append('background solid white')

    commands_chimera.append('color dark gray #')
    for k,chain_ in enumerate(list_chains):
        commands_chimera.append('color %s #0.%s:.%s'%(chain_colors[k % len(chain_colors) ], model, chain_) )


    for k,sector in enumerate(sectors):
        sector_pdb = mapping[sector]
        sector_mapped = sector[sector_pdb>=0]
        sector_pdb_mapped = sector_pdb[sector_pdb>=0]
        count = 0
        if simultaneous:
            sector_color = sector_colors[k]
        else:
            sector_color = sector_colors[0]
        if with_numbers:
            l_original_previous = -100
        for l_original,l in zip(sector_mapped,sector_pdb_mapped):
            commands_chimera.append(
            'color %s #0.%s:%s.%s'% ( sector_color, model, index_pdb[l][1],index_pdb[l][0]) )
            if show_sidechains:
                commands_chimera.append(
                'display #0.%s:%s.%s'% ( model, index_pdb[l][1],index_pdb[l][0]) )
            if with_numbers:
                if l_original >= l_original_previous + with_numbers_every:
                    l_original_previous = copy.copy(l_original)
                    if pdb_numbers:
                        label = index_pdb[l][1]
                    else:
                        label = l_original+1
                    commands_chimera.append(
                    "setattr r label '%s' #0.%s:%s.%s"%(label, model, index_pdb[l][1],index_pdb[l][0]) )
                    commands_chimera.append(
                    "color black,l #0.%s:%s.%s"%(model, index_pdb[l][1],index_pdb[l][0]) )
            count +=1

        if not simultaneous:
            commands_chimera.append('copy file %s/sector_%s.png png width 2000 height 2000'%(output_folder,sector_names[k]) )
            if k != len(sectors)-1:
                # Recolor back to previous.
                commands_chimera.append('color dark gray #')
                for k,chain_ in enumerate(list_chains):
                    commands_chimera.append('color %s #0.%s:.%s'%(chain_colors[k % len(chain_colors) ], model, chain_) )

                # Remove all labels.
                commands_chimera.append('~rlabel')
                commands_chimera.append('~show')


    if simultaneous:
        commands_chimera.append('copy file %s/all_sectors.png png width 2000 height 2000'%output_folder)

    if exit:
        commands_chimera.append('stop confirmed')


    with open('%s/chimera_commands.py'%output_folder,'w') as f:
        f.write('import chimera\n')
        f.write('from chimera import runCommand\n')
        for command in commands_chimera:
            f.write('runCommand("%s")\n'%command)


    os.system('%s --script %s/chimera_commands.py'%(chimera_path,output_folder) )
    return
