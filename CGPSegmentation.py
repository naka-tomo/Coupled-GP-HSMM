# encoding: utf8
from __future__ import unicode_literals
from __future__ import print_function
import GaussianProcessMultiDim
import random
import math
import matplotlib.pyplot as plt
import time
import numpy as np
import sys
import os


class CGPSegmentation():
    # parameters
    MAX_LEN = 13
    MIN_LEN = 5
    AVE_LEN = 7
    SKIP_LEN = 1
    LAG_SIG = 1     # 時間差の分散

    def __init__(self, dim, nclass):
        self.dim = dim
        self.numclass = nclass
        self.segmlen = 3
        self.gps = [ GaussianProcessMultiDim.GPMD(dim) for i in range(self.numclass) ]
        self.segm_in_class= [ [] for i in range(self.numclass) ]
        self.segmclass = {}
        self.segments = []
        self.trans_prob = np.ones( (nclass,nclass) )
        self.trans_prob_bos = np.ones( nclass )
        self.trans_prob_eos = np.ones( nclass )
        self.is_initialized = False

        
        # 時間遅れに関する変数
        self.lag_range = [-4,-3,-2,-1,0,1,2,3,4]    # ズレの推定範囲
        self.lag_probs = [ math.exp(-d*d/(2*self.LAG_SIG*self.LAG_SIG)) for d in self.lag_range ]  # ズレの確率
        self.t_lag_pairs = [] # 時刻tの自身の行動と対応する相手の行動の時間差
        self.interrel_prob = np.zeros((nclass, nclass)) # 行動の相互関係を表現する確率
        self.opponent_gp = None  # 相手のGP

        
    def load_data(self, filenames, classfile=None ):
        self.data = []
        self.segments = []
        self.is_initialized = False

        for fname in filenames:
            y = np.loadtxt( fname )
            segm = []
            self.data.append( y )


            # ランダムに切る
            i = 0
            while i<len(y):
                length = random.randint(self.MIN_LEN, self.MAX_LEN)

                if i+length+1>=len(y):
                    length = len(y)-i

                segm.append( y[i:i+length+1] )

                i+=length

            self.segments.append( segm )

            # ランダムに割り振る
            for i,s in enumerate(segm):
                c = random.randint(0,self.numclass-1)
                self.segmclass[id(s) ] = c

        # 遷移確率更新
        self.calc_trans_prob()
        self.own_classes = self.get_classes_list()

        self.t_lag_pairs = [ [] for i in range(len(self.data)) ]



    def load_model( self, basename ):
        # GP読み込み
        for c in range(self.numclass):
            filename = basename + "class%03d.npy" % c
            self.segm_in_class[c] = np.load( filename )
            self.update_gp( c )

        # 遷移確率更新
        self.trans_prob = np.load( basename+"trans.npy" )
        self.trans_prob_bos = np.load( basename+"trans_bos.npy" )
        self.trans_prob_eos = np.load( basename+"trans_eos.npy" )


    def update_gp(self, c ):
        datay = []
        datax = []
        for s in self.segm_in_class[c]:
            datay += [ y for y in s ]
            datax += range(len(s))
            
        # 相手のデータも加えて学習
        for s in self.opponent_gp.segm_in_class[c]:
            datay += [ y for y in s ]
            datax += range(len(s)) 

        self.gps[c].learn( datax, datay )


    def calc_emission_prob( self, c, segm ):
        gp = self.gps[c]
        slen = len(segm)

        if len(segm) > 2:
            plen = self.AVE_LEN**slen * math.exp(-self.AVE_LEN) / math.factorial(slen)
            p = gp.calc_lik( np.arange(len(segm), dtype=np.float) , segm )
            return math.exp(p) * plen
        else:
            return 0

    def save_model(self, basename ):
        if not os.path.exists(basename):
            os.mkdir( basename )

        for n,segm in enumerate(self.segments):
            classes = []
            cut_points = []
            for s in segm:
                c = self.segmclass[id(s)]
                classes += [ c for i in range(len(s)) ]
                cut_points += [0] * len(s)
                cut_points[-1] = 1
            np.savetxt( basename+"segm%03d.txt" % n, np.vstack([classes,cut_points]).T, fmt=str("%d") )


        # 各クラスに分類されたデータを保存
        for c in range(len(self.gps)):
            for d in range(self.dim):
                plt.clf()
                for data in self.segm_in_class[c]:
                    if self.dim==1:
                        plt.plot( range(len(data)), data, "o-" )
                    else:
                        plt.plot( range(len(data[:,d])), data[:,d], "o-" )
                    plt.ylim( -1, 1 )
                plt.savefig( basename+"class%03d_dim%03d.png" % (c, d) )
                
        # 相互関係確率を保存
        np.save( basename + "interrelationship_prob.npy", self.interrel_prob )
        
        # 時間差を保存
        for i in range(len(self.data)):
            lags = np.zeros( len(self.data[i]) )
            for t, lag in self.t_lag_pairs[i]:
                lags[t] = lag
            np.savetxt( basename+"lag%03d.txt"%i, lags, fmt="%d" )

        # テキストでも保存
        np.save( basename + "trans.npy" , self.trans_prob  )
        np.save( basename + "trans_bos.npy" , self.trans_prob_bos )
        np.save( basename + "trans_eos.npy" , self.trans_prob_eos )

        for c in range(self.numclass):
            np.save( basename+"class%03d.npy" % c, self.segm_in_class[c] )


    def forward_filtering(self, idx ):
        d = self.data[idx]
        T = len(d)
        a = np.zeros( (len(d), self.MAX_LEN, self.numclass, len(self.lag_range)) )   # 前向き確率
        z = np.ones( T ) # 正規化定数
        
        # 相手のクラスを取得
        oppnect_classes = self.opponent_gp.get_classes_list()[idx]

        for t in range(T):
            for k in range(self.MIN_LEN,self.MAX_LEN,self.SKIP_LEN):
                if t-k<0:
                    break
            

                segm = d[t-k:t+1]
                for c in range(self.numclass):
                    out_prob = self.calc_emission_prob( c, segm )
                    foward_prob = 0.0
                    

                    # 遷移確率
                    tt = t-k-1
                    if tt>=0:
                        #for kk in range(self.MAX_LEN):
                        #    for cc in range(self.numclass):
                        #        for ll in range(len(self.lag_range)):
                        #            foward_prob += a[tt,kk,cc,ll] * self.trans_prob[cc, c] * z[tt]
                        # t-k-1を終端とし多分節の可能性をすべて周辺化
                        foward_prob = np.sum( np.sum(a[tt,:,:,:],2) * self.trans_prob[:,c] ) * out_prob * z[tt]
                    else:
                        # 最初の単語
                        foward_prob = out_prob * self.trans_prob_bos[c]

                    if t==T-1:
                        # 最後の単語
                        foward_prob *= self.trans_prob_eos[c]
    
                    # 時間差に関する確率
                    for l in range(len(self.lag_range)):
                        lag = self.lag_range[l]
                        
                        if t+lag<0 or t+lag>=T:
                            break
                        
                        oc = oppnect_classes[t+lag]
                        a[t,k,c,l] = foward_prob * self.interrel_prob[oc,c] * self.lag_probs[l]
                        if math.isnan(foward_prob):
                            print( "a[t=%d,k=%d,c=%d] became NAN!!" % (t,k,c) )
                            sys.exit(-1)
            # 正規化
            if t-self.MIN_LEN>=0:
                z[t] = np.sum( a[t,:,:,:] )
                a[t,:,:,:] /= z[t]
                 
        return a

    def sample_idx(self, prob ):
        accm_prob = [0,] * len(prob)
        for i in range(len(prob)):
            accm_prob[i] = prob[i] + accm_prob[i-1]

        rnd = random.random() * accm_prob[-1]
        for i in range(len(prob)):
            if rnd <= accm_prob[i]:
                return i


    def backward_sampling(self, a, d, use_max):
        T = a.shape[0]
        t = T-1
        L = len(self.lag_range)

        segm = []
        segm_class = []
        t_lag_pairs = []

        while True:
            if not use_max:
                # サンプリングする                
                idx = self.sample_idx( a[t].reshape( self.MAX_LEN*self.numclass*L ))
            else:                
                # サンプリングはせずに確率の最大値を使う
                idx = np.argmax( a[t].reshape( self.MAX_LEN*self.numclass*L ) )

            #k = int(idx/self.numclass)
            #c = idx % self.numclass
            
            
            k = int(idx/(self.numclass*L))
            c = int((idx % (self.numclass*L))/L)
            l = (idx % (self.numclass*L)) % L
                        

            s = d[t-k:t+1]

            segm.insert( 0, s )
            segm_class.insert( 0, c )
            t_lag_pairs.append( (t, self.lag_range[l]) )

            t = t-k-1

            if t<=0:
                break

        return segm, segm_class, t_lag_pairs

    def calc_trans_prob( self ):
        self.trans_prob = np.zeros( (self.numclass,self.numclass) )
        self.trans_prob += 0.1

        # 数え上げる
        for n,segm in enumerate(self.segments):
            for i in range(1,len(segm)):
                try:
                    cc = self.segmclass[ id(segm[i-1]) ]
                    c = self.segmclass[ id(segm[i]) ]
                except KeyError as e:
                    # gibss samplingで除かれているものは無視
                    break
                self.trans_prob[cc,c] += 1.0

        # 正規化
        self.trans_prob = self.trans_prob / self.trans_prob.sum(1).reshape(self.numclass,1)

    # excluded_idxを除いた残りで，相互関係確率を計算
    def calc_interrel_prob(self, excluded_idx ):
        trans_count = np.zeros( (self.numclass, self.numclass) ) + 0.1
        own_classes = self.get_classes_list()
        opponent_classes = self.opponent_gp.get_classes_list()
        for i in range(len(self.data)):
            if i!=excluded_idx:
                for t, lag in self.t_lag_pairs[i]:
                    c1 = opponent_classes[i][t+lag]
                    c2 = own_classes[i][t]
                    trans_count[c1,c2] += 1

        self.interrel_prob = trans_count / trans_count.sum(1).reshape(self.numclass,1)
                    

    # list.remove( elem )だとValueErrorになる
    def remove_ndarray(self, lst, elem ):
        l = len(elem)
        for i,e in enumerate(lst):
            if len(e)!=l:
                continue
            if (e==elem).all():
                lst.pop(i)
                return
        raise ValueError( "ndarray is not found!!" )

    def learn(self, opponent_gp, use_max=False ):
        # use_max: サンプリングをせずに，最大値を選択
        
        # 相手のクラス
        self.opponent_gp = opponent_gp
        if self.is_initialized==False:
            # GPの学習
            for i in range(len(self.segments)):
                for s in self.segments[i]:
                    c = self.segmclass[id(s)]
                    self.segm_in_class[c].append( s )

            # 各クラス毎に学習
            for c in range(self.numclass):
                self.update_gp( c )

            self.is_initialized = True

        self.update(True, use_max)

    def recog(self):
        self.update(False)

    def update(self, learning_phase=True, use_max=False ):

        for i in range(len(self.segments)):
            d = self.data[i]
            segm = self.segments[i]

            for s in segm:
                c = self.segmclass[id(s)]
                self.segmclass.pop( id(s) )

                if learning_phase:
                    # パラメータ更新
                    self.remove_ndarray( self.segm_in_class[c], s )

            if learning_phase:
                # GP更新
                for c in range(self.numclass):
                    self.update_gp( c )

                # 遷移確率更新
                self.calc_trans_prob()
                self.calc_interrel_prob(i)

            start = time.clock()
            print( "forward...", end="")
            a = self.forward_filtering( i )

            print( "backward...", end="" )
            segm, segm_class, t_lag_pairs = self.backward_sampling( a, d, use_max )
            print( time.clock()-start, "sec" )

            print( "Number of classified segments: [", end="")
            for s in self.segm_in_class:
                print( len(s), end=" " )
            print( "]" )


            self.segments[i] = segm
            self.t_lag_pairs[i] = t_lag_pairs

            for s,c in zip( segm, segm_class ):
                self.segmclass[id(s)] = c

                # パラメータ更新
                if learning_phase:
                    self.segm_in_class[c].append(s)

            if learning_phase:
                # GP更新
                for c in range(self.numclass):
                    self.update_gp( c )

                # 遷移確率更新
                self.calc_trans_prob()
                self.calc_interrel_prob(-1)


        return

    def calc_lik(self):
        lik = 0
        for segm in self.segments:
            for s in segm:
                c = self.segmclass[id(s)]
                #lik += self.gps[c].calc_lik( np.arange(len(s),dtype=np.float) , np.array(s) )
                lik += self.gps[c].calc_lik( np.arange(len(s), dtype=np.float) , s )

        return lik


    # クラスを一列に並べた配列を返す
    def get_classes_list(self):
        classes_list = []
        for n,segm in enumerate(self.segments):
            classes = []
            for s in segm:
                if id(s) in self.segmclass:
                    c = self.segmclass[id(s)]
                    classes += [ c for i in range(len(s)) ]
            classes_list.append( classes )
        return classes_list

