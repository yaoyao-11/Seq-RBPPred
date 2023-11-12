#! /usr/local/env python

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import csv
import tensorflow as tf
import numpy as np
import subprocess
import glob
from Bio import SeqIO
from unirep import babbler1900 as babbler

def fe1900(seq, model):
    if not model.is_valid_seq(seq):
        return None
    features = model.get_rep(seq)
    return features

def batch_embedding(fanamelist, output_prefix):
    MAXLEN = 3000
    for faname in fanamelist:
        recordnames = []
        o1 = o2 = o3 = np.arange(1900)
        i = 0
        for i, record in enumerate(SeqIO.parse(faname, "fasta")):
            res = fe1900(str(record.seq), b)
            if (res is None):
                print(record.id, " Failed!")
                continue
            t1, t2, t3 = res
            o1 = np.row_stack((o1, t1))
            o2 = np.row_stack((o2, t2))
            o3 = np.row_stack((o3, t3))
            recordnames.append(record.id)
            print(record.id + "  OK!")
            print("output shape: " + str(o1.shape[0]))

            if (i % MAXLEN == 0 and i != 0):
                n = i // MAXLEN - 1
                np.savetxt(os.path.join(output_prefix, faname + ".avghid.csv." + str(n)), o1[1:], delimiter=',')
                np.savetxt(os.path.join(output_prefix, faname + ".finhid.csv." + str(n)), o2[1:], delimiter=',')
                np.savetxt(os.path.join(output_prefix, faname + ".fincel.csv." + str(n)), o3[1:], delimiter=',')
                with open(os.path.join(output_prefix, faname + ".rowname.csv." + str(n)), 'w', newline='') as f:
                    cf = csv.writer(f, delimiter='\n')
                    cf.writerow(recordnames)
                    f.seek(f.tell()-2, os.SEEK_SET)
                    f.truncate()
                recordnames = []
                o1 = o2 = o3 = np.arange(1900)

        if (i % MAXLEN != 0):
            n = i // MAXLEN
            np.savetxt(os.path.join(output_prefix, faname + ".avghid.csv." + str(n)), o1[1:], delimiter=',')
            np.savetxt(os.path.join(output_prefix, faname + ".finhid.csv." + str(n)), o2[1:], delimiter=',')
            np.savetxt(os.path.join(output_prefix, faname + ".fincel.csv." + str(n)), o3[1:], delimiter=',')
            with open(os.path.join(output_prefix, faname + ".rowname.csv." + str(n)), 'w', newline='') as f:
                cf = csv.writer(f, delimiter='\n')
                cf.writerow(recordnames)
                f.seek(f.tell()-2, os.SEEK_SET)
                f.truncate()


def rowname_concat(pathname, filepatterns):
    outpath = "../output/"
    if (not os.path.exists(os.path.join(pathname, outpath))):
        os.mkdir(os.path.join(pathname, outpath))
    
    

    rownamefiles = glob.glob(os.path.join(pathname, filepatterns[0]))
    for i in rownamefiles:
        subprocess.run("""
            paste -d',' {i} {avghid} >   {avghid}.f 
            paste -d',' {i} {finhid} >   {finhid}.f
            paste -d',' {i} {fincel} >   {fincel}.f
        """.format(i=i, avghid= i.replace("rowname", "avghid"), 
                    finhid= i.replace("rowname", "finhid"), fincel = i.replace("rowname", "fincel")), 
        shell=True, check=True, stdout=subprocess.PIPE)


    ahfiles = [n[:-2] for n in glob.glob(os.path.join(pathname, filepatterns[1]))]
    fhfiles = [n[:-2] for n in glob.glob(os.path.join(pathname, filepatterns[2]))]
    fcfiles = [n[:-2] for n in glob.glob(os.path.join(pathname, filepatterns[3]))]

    for ah, fh, fc in zip(ahfiles, fhfiles, fcfiles):
        ahs = ah.split('/')
        fhs = fh.split('/')
        fcs = fc.split('/')
        ahs.insert(-1, outpath)
        fhs.insert(-1, outpath)
        fcs.insert(-1, outpath)

        subprocess.run("""
            cat {oavghid}.*.f > {avghid}.fin
            cat {ofinhid}.*.f > {finhid}.fin
            cat {ofincel}.*.f > {fincel}.fin
        """.format(oavghid=ah, ofinhid=fh, ofincel=fc,
                    avghid='/'.join(ahs), 
                    finhid='/'.join(fhs), 
                    fincel='/'.join(fcs)), 
                    shell=True, check=True, stdout=subprocess.PIPE)

if __name__ == "__main__":

    MODEL_WEIGHT_PATH = "./1900_weights"
    batch_size = 6
    dirname = "data"
    b = babbler(batch_size=batch_size, model_path=MODEL_WEIGHT_PATH)
    files = [os.path.join(dirname, fn) for fn in os.listdir(dirname)]
    batch_embedding(files, "./")
    rowname_concat(dirname, ["*.rowname.csv.*", "*.avghid.csv.0", "*.finhid.csv.0", "*.fincel.csv.0"])