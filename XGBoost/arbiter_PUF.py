# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 23:40:37 2021

@author: weber
"""
import pypuf.simulation
import pypuf.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import array
from LFSR_simulated import*
from Puf_resilience import*

class arbiter_PUF:
    def __init__(self):
        self.LFSR_simulated = LFSR_simulated()
        self.Puf_resilience = Puf_resilience()
    
    def total_delay_diff(self, challenge, puf):
        challenge = array([challenge])
        last_stage_ind = len(challenge[0])-1
        puf_delay = pypuf.simulation.LTFArray(weight_array=puf.weight_array[:, :64], bias=None, transform=puf.transform)
        stage_delay_diff = puf_delay.val(challenge[:, :64])
        #print("additive delay",stage_delay_diff)
        return stage_delay_diff

    def load_data(self, stages, data_num, puf_seed, cus_seed, base):
        puf = pypuf.simulation.ArbiterPUF(n=(stages-4), seed=puf_seed)
        #puf = pypuf.simulation.ArbiterPUF(n=(stages-4), seed=12, noisiness=.05)
        lfsrChallenges = random_inputs(n=stages, N=data_num, seed=cus_seed) # LFSR random challenges data
        train_data = []
        train_label = []
        data = []
        data_label = []
        delay_diff = []
        qcut_one_hot = []
        
        test_crps = lfsrChallenges
        
        for i in range(data_num):
            ### data ###
            challenge = test_crps[i]
            
            # obfuscate part
            obfuscateChallenge = self.LFSR_simulated.createObfuscateChallenge(challenge, base)
            obfuscateChallenge = [-1 if c == 0 else c for c in obfuscateChallenge]
            #obfuscateChallenge = self.Puf_resilience.cyclic_shift(challenge, puf)
            #obfuscateChallenge = [-1 if c == 0 else c for c in obfuscateChallenge]
            
            #final_delay_diff = puf.val(np.array([obfuscateChallenge]))
            #final_delay_diff = self.total_delay_diff(challenge[4:], puf)
            
            challenge = challenge[4:]
            final_delay_diff = self.total_delay_diff(challenge, puf)
            #challenge = [0 if c == -1 else c for c in challenge]       
                  
            response = self.LFSR_simulated.produceObfuscateResponse(puf, obfuscateChallenge)
            response = np.array(response)
            data_r = -1
            if response == -1:
                data_r = -1
            else:
                data_r = 1
            #print("dd")
            #print(challenge)
            data.append(challenge)
            delay_diff.append(final_delay_diff[0])
            data_label.append(data_r)
            #print("data label", data_r)
          
        data = np.array(data)
        qcut_label = pd.qcut(delay_diff, q=4, labels=["1", "2", "3", "4"])
        
        data_cut = []
        for x in range(len(qcut_label)):
           if qcut_label[x] == "1":
               data_cut.append(np.concatenate((data[x],[1,0,0,0])))
               #data_cut.append([1,0,0,0])
           elif qcut_label[x] == "2":
               data_cut.append(np.concatenate((data[x],[0,1,0,0])))
               #data_cut.append([0,1,0,0])
           elif qcut_label[x] == "3":
               data_cut.append(np.concatenate((data[x],[0,0,1,0])))
               #data_cut.append([0,0,1,0])
           else:
               data_cut.append(np.concatenate((data[x],[0,0,0,1])))
               #data_cut.append([0,0,0,1])
        
        data_cut = np.array(data_cut)
        train_data = data_cut
        train_label = np.array(data_label)
        
        # for pypuf lr2021 attack
        # attack_info = np.array(data)
        
        return train_data, train_label
        # return train_data, train_label, attack_info
    
    
    def load_data_2021(self, stages, data_num, puf_seed, cus_seed, base, layer_ouput):
        puf = pypuf.simulation.ArbiterPUF(n=(stages-4), seed=puf_seed)
        #puf = pypuf.simulation.ArbiterPUF(n=(stages-4), seed=12, noisiness=.05)
        lfsrChallenges = random_inputs(n=stages, N=data_num, seed=cus_seed) # LFSR random challenges data
        train_data = []
        train_label = []
        data = []
        data_label = []
        delay_diff = []
        qcut_one_hot = []
        
        test_crps = lfsrChallenges
        
        for i in range(data_num):
            ### data ###
            challenge = test_crps[i]
            
            # obfuscate part
            obfuscateChallenge = self.LFSR_simulated.createObfuscateChallenge(challenge, base)
            obfuscateChallenge = [-1 if c == 0 else c for c in obfuscateChallenge]
            #obfuscateChallenge = self.Puf_resilience.cyclic_shift(challenge, puf)
            #obfuscateChallenge = [-1 if c == 0 else c for c in obfuscateChallenge]
            
            #final_delay_diff = puf.val(np.array([obfuscateChallenge]))
            #final_delay_diff = self.total_delay_diff(challenge[4:], puf)
            
            challenge = challenge[4:]
            final_delay_diff = layer_ouput[i]
            #challenge = [0 if c == -1 else c for c in challenge]       
                  
            response = self.LFSR_simulated.produceObfuscateResponse(puf, obfuscateChallenge)
            response = np.array(response)
            data_r = -1
            if response == -1:
                data_r = -1
            else:
                data_r = 1
            #print("dd")
            #print(challenge)
            data.append(challenge)
            delay_diff.append(final_delay_diff[0])
            data_label.append(data_r)
            #print("data label", data_r)
          
        data = np.array(data)
        qcut_label = pd.qcut(delay_diff, q=4, labels=["1", "2", "3", "4"])
        
        data_cut = []
        for x in range(len(qcut_label)):
           if qcut_label[x] == "1":
               data_cut.append(np.concatenate((data[x],[1,0,0,0])))
               #data_cut.append([1,0,0,0])
           elif qcut_label[x] == "2":
               data_cut.append(np.concatenate((data[x],[0,1,0,0])))
               #data_cut.append([0,1,0,0])
           elif qcut_label[x] == "3":
               data_cut.append(np.concatenate((data[x],[0,0,1,0])))
               #data_cut.append([0,0,1,0])
           else:
               data_cut.append(np.concatenate((data[x],[0,0,0,1])))
               #data_cut.append([0,0,0,1])
        
        data_cut = np.array(data_cut)
        train_data = data_cut
        train_label = np.array(data_label)
        
        # for pypuf lr2021 attack
        # attack_info = np.array(data)
        # return train_data, train_label
        return train_data, train_label, attack_info
        