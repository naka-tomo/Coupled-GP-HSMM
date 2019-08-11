# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import print_function
from CGPSegmentation import CGPSegmentation
import time
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_result( savedir, ndata ):
    for n in range(ndata):
        classes_a = np.loadtxt( os.path.join( savedir, "A", "segm%03d.txt"%n ) )[:,0]
        classes_b = np.loadtxt( os.path.join( savedir, "B", "segm%03d.txt"%n ) )[:,0]
        lags = np.loadtxt( os.path.join( savedir, "A", "lag%03d.txt"%n ) )
    
        plt.figure()
        plt.plot( range(len(classes_a)), classes_a, "o" )
        plt.plot( range(len(classes_b)), classes_b, "x" )
        
        for t in range(len(classes_a)):
            lag = int(lags[t])
            
            if lag!=0:
                c1 = classes_a[t]
                c2 = classes_b[t+lag]                
                plt.plot( [t,t+lag], [c1,c2], "k-" )
        
        plt.savefig( os.path.join( savedir, "result%03d.png" % n ) )
        
    #plt.show()
        

def learn_cgphsmm( savedir ):
    
    if not os.path.exists(savedir):
        os.mkdir( savedir )
    
    gp_a = CGPSegmentation(2,5)
    gp_b = CGPSegmentation(2,5)

    files_a =  [ "a%03d.txt" % j for j in range(8) ]
    files_b =  [ "b%03d.txt" % j for j in range(8) ]
    gp_a.load_data( files_a )
    gp_b.load_data( files_b )

    start = time.clock()
    for it in range(5):
        # 最後だけ確率の最大値を使う
        if it==4:
            use_max=True
        else:
            use_max=False
            
        print( "*****", it, "*****" )

        # Aの学習
        print( "--- A" )
        gp_a.learn( gp_b, use_max )
        gp_a.save_model( os.path.join(savedir, "A/") )
        print( "lik", gp_a.calc_lik() )

        # Bの学習        
        print( "--- B" )
        gp_b.learn( gp_a, use_max )
        gp_b.save_model( os.path.join(savedir, "B/") )
        print( "lik", gp_b.calc_lik() )
    
        
    print( time.clock()-start )



def main():
    learn_cgphsmm( "learn/" )
    plot_result( "learn", 8 )
    return

if __name__=="__main__":
    main()