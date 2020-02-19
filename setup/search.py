import numpy as np
from numpy import linalg
import pyrr
import sys
from Geometries import Geometries
from Geometries import mat_from_fields
from Evaluation import Evaluation
from Evaluation import EvaluationOne
from Evaluation import EvaluationSlides
import math
import random
import argparse
from itertools import product
from datetime import datetime
import os


def mat_from_field(hf,ax,n0,n1):
    dim = len(hf)
    mat = np.zeros(shape=(dim,dim,dim))
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                ind = [i,j]
                ind3d = ind.copy()
                ind3d.insert(ax,k)
                ind3d = tuple(ind3d)
                ind2d = tuple(ind)
                if k<hf[ind2d]:
                    mat[ind3d]=n0
                else: mat[ind3d]=n1
    mat = np.array(mat)
    return mat

def first_hf(fixed_sides,sax,noc,dim):
    print("Running computational search for valid joint geometries...")
    start_time = datetime.now()
    # Prepare to get all combinations for one height field
    hlist = []
    for h in range(dim+1): hlist.append(h)

    # Browse all timbers
    for n in range(noc):

        # Browse fabrication direction(s)
        fst,fen = 0,2
        if n==0: fen=1
        if n==noc-1: fst=1
        for m in range(fst,fen):

            start_time_loc = datetime.now()
            # Initiate failure mode counters
            uncon_cnt = ouncon_cnt = mslid_cnt = frag_cnt = valid_cnt = num_cnt = total_cnt = 0

            # Browse all geometry combinations
            roll = product(hlist, repeat=dim*dim)
            for i,flat_hf in enumerate(list(roll)):

                ### while testing ###
                #if total_cnt>999: break
                ###

                # Total count
                total_cnt +=1
                if total_cnt%10000==0:
                    print(total_cnt,"...", datetime.now()-start_time)

                # Check so that there are not too few or to many voxels in this geometry
                sum = np.sum(np.array(flat_hf))
                min_vox = 3
                if sum<=min_vox or sum>=(dim*dim*dim-(noc-1)*min_vox):
                     num_cnt+=1
                     continue

                # If it is a middle timber, check so that there is at least one "opening"
                if n!=0 and n!=noc-1:
                    if np.sum(np.array(flat_hf)==0)==0:
                        continue

                # Translate flat field to 2d matrix
                hf = np.zeros((dim,dim))
                for i_ in range(dim):
                    for j_ in range(dim):
                        hf[i_][j_] = list(flat_hf)[i_*dim+j_]

                # Translate 2d matrix to 3d matrix
                n0, n1 = n, 0
                if n0==0: n1 = 1
                if m==1: n0,n1 = n1,n0
                voxel_matrix =  mat_from_field(hf,sax,n0,n1)

                # Evaluate
                eval = EvaluationOne(voxel_matrix,fixed_sides,sax,noc,n,False)
                # Count
                if   not eval.connected_and_bridged: uncon_cnt+=1
                elif not eval.other_connected_and_bridged: ouncon_cnt+=1
                elif not eval.interlock: mslid_cnt+=1
                elif not eval.nofragile: frag_cnt+=1
                else:
                    valid_cnt+=1
                    if m==0: np.save("data/3/20_10_00/"+str(n+1)+"a/height_field_"+str(valid_cnt)+".npy",hf)
                    else: np.save("data/3/20_10_00/"+str(n+1)+"b/height_field_"+str(valid_cnt)+".npy",hf)
            print("\n-----Timber number",n+1,", Fabrication direction",m,"-----")
            print("Total:\t\t",   total_cnt)
            print("Too few/many:\t",num_cnt)
            print("Unconn/unbri:\t",  uncon_cnt)
            print("Other uncon..:\t",  ouncon_cnt)
            print("Mult slides:\t",  mslid_cnt)
            print("Fragile parts:\t",frag_cnt)
            print("Valid:\t\t",      valid_cnt)
            print("\nSearch finished in",datetime.now()-start_time_loc)
    print("\nAll searches finished in",datetime.now()-start_time)

def second_hfs(fixed_sides,sax,noc,dim):
    print("Creating following height fields from first...")
    start_time = datetime.now()
    hlist = []
    cnt = 0
    for h in range(dim+1): hlist.append(h)

    # Browse files of height fields
    files = os.listdir("data/3/20_10_00/2a")
    for fi,filename in enumerate(files):
        if fi%10==0: print(cnt,fi,datetime.now()-start_time)

        # Second hf
        hf2 = np.load("data/3/20_10_00/2a/"+filename)

        # First hf base
        hf1base = np.zeros((dim,dim))
        for i in range(dim):
            for j in range(dim):
                if hf2[i,j]==0: hf1base[i,j]=99
                else: hf1base[i,j]=0

        # Combinations for first hf by roll
        rollcnt = (hf2==0).sum()
        #print("roll count",rollcnt)
        if rollcnt!=0:
            roll = product(hlist, repeat=rollcnt)
            for values in roll:
                hf1 = np.copy(hf1base)
                inds = np.argwhere(hf1==99)
                for ind,val in zip(inds,values):
                    hf1[tuple(ind)]=val
                # Make sure there are at least 3 voxels for each component
                sum = np.sum(np.array(hf1))
                min_vox = 3
                if sum<=min_vox or sum>=(dim*dim*dim-(noc-1)*min_vox): continue
                # Evaluate the timber with 2nd fewest solutions (2)
                voxel_matrix =  mat_from_fields([hf1,hf2],sax)
                eval = EvaluationOne(voxel_matrix,fixed_sides,sax,noc,2,True)
                if eval.valid:
                    # Evaluate the last timber with most solutions (0)
                    #eval = EvaluationOne(voxel_matrix,fixed_sides,sax,noc,0,True)
                    #if eval.valid:
                    np.save("data/3/20_10_00/2a_full/height_fields_"+str(cnt)+".npy",[hf1,hf2])
                    cnt+=1
                    if cnt%10==0: print(cnt,fi,datetime.now()-start_time)
    print("Created",cnt-1,"new geometries from",len(files),"initiations in",datetime.now()-start_time)

def reduce(fixed_sides,sax,noc,dim):
    print("Order valid results according to sliding depth product...")
    start_time = datetime.now()

    slide_depts = []
    HF = []

    # Browse files of height fields
    files = os.listdir("data/3/20_10_00/2a_full")
    for fi,filename in enumerate(files):
        if fi%100==0: print(fi,"...")

        # Load heightfileds
        hfs = np.load("data/3/20_10_00/2a_full/"+filename)
        HF.append(hfs)

        # Get voxel matrix
        voxel_matrix =  mat_from_fields(hfs,sax)

        # Evaluate sliding direction
        eval = EvaluationSlides(voxel_matrix,fixed_sides,sax,noc)
        slide_depts.append(eval.slide_depths)

    slide_depts = np.sort(slide_depts)

    for j in range(noc):
        max_dep = np.amax(slide_depts, axis=0)[j]
        inds = []
        for i in range(len(slide_depts)):
            if slide_depts[i][j]!=max_dep: inds.append(i)
        slide_depts = np.delete(slide_depts, inds, 0)
        HF = np.delete(HF, inds, 0)

    for i in range(len(HF)): np.save("data/3/20_10_00/2a_rank/height_fields_"+str(i)+".npy",HF[i])

    print("Reduced",len(files),"geometries to", len(HF),"in",datetime.now()-start_time)

if __name__ == "__main__":

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", action='store_true')
    parser.add_argument("--second", action='store_true')
    parser.add_argument("--reduce", action='store_true')
    args = parser.parse_args()

    # Shared variables
    fixed_sides = [[[2,0]],[[1,0]],[[0,0]]]
    sax = 2
    noc = 3
    dim = 3

    if args.first: first_hf(fixed_sides,sax,noc,dim)
    if args.second: second_hfs(fixed_sides,sax,noc,dim)
    if args.reduce: reduce(fixed_sides,sax,noc,dim)
