from numpy.core.fromnumeric import size
import pandas as pd
import numpy as np
import torch
#file = "conceptnet-5.7.0-rel.csv"
def conceptnet_to_dict(csv_path):
    data = pd.read_csv(csv_path, delimiter=',')
    data['start'] = data['start'].apply(lambda str: str.split("en/")[1].split("/")[0]) 
    data['end'] = data['end'].apply(lambda str: str.split("en/")[1].split("/")[0]) 
    data['relation'] = data['relation'].apply(lambda str: str.split('/r/')[-1])
    data['start'] = data['start'].apply(lambda str: str.lower())
    data['end'] = data['end'].apply(lambda str: str.lower())
    data['relation'] = data['relation'].apply(lambda str: str.lower())
    net_dict = {}
    for (a,b,c) in zip(data['start'], data['end'], data['relation']):        
        if a not in net_dict:
            #net_dict[(a, b)] = [set([i]), set([c])]
            net_dict[a] = {}
            net_dict[a][b] = set([c])
        elif b not in net_dict[a]:
            #net_dict[(a, b)][0].add(i)
            #net_dict[(a, b)][1].add(c) 
            net_dict[a][b] = set([c])   
        else:   
            net_dict[a][b].add(c)    
    for x in net_dict:
        for i, j in net_dict[x].items():
            net_dict[x][i] = len(j)
    return net_dict

def cal_path_reliability(net_dict, cached_dict, key, init_resource = 1, step = 2, allow_loop = True): #BFS algorithm
    #print("in cal",net_dict)
    flag_set = set() #bfs'flags
    if not allow_loop: flag_set.add(key) 
    if key not in cached_dict: cached_dict[key] = {}
    cached_dict[key][key] = init_resource
    query = set([key])
    bfs_next = set()
    while step: 
        for q in query:  
            distr = 0 
            next_step = set() 
            if q not in net_dict: continue            
            for word, paths in net_dict[q].items():
                if not allow_loop and word in flag_set : continue
                else: 
                    distr += paths
                    next_step.add(word)

            resource = cached_dict[key][q]
            distrib_res = resource / distr if distr else 0

            for word in next_step:
                if word not in cached_dict[key]:
                    cached_dict[key][word] = distrib_res * net_dict[q][word]
                else: cached_dict[key][word] += distrib_res * net_dict[q][word]
                #print("sss: ",q,word,next_step, cached_dict)
            bfs_next = bfs_next.union(next_step)
        #print("/n in bfs, step", 2-step, query, bfs_next, flag_set, cached_dict)
        if not allow_loop: flag_set = flag_set.union(bfs_next)
        query.clear()
        query = query.union(bfs_next)
        bfs_next.clear()
        step -= 1
            


def point2point_reliability(net_dict, cached_dict, query_text, key_text, query_map, key_map, qcon, kcon, steps=2, allow_loop=True,):
    # text (B, text)
    max_len = 0
    kg_score = np.zeros((len(query_map), len(key_map)))
    i,j = 0,0
    for qword in query_text:
        j = 0
        #print(cached_dict)
        for kword in key_text:
            #print(qword, kword)
            
            if kword in net_dict:
                if qword not in cached_dict:
                    cal_path_reliability(net_dict, cached_dict, qword, step=steps, allow_loop=allow_loop)
                    #input()
                flag = key_map[j]
                while(j < len(key_map) and flag == key_map[j]):
                    if kword in cached_dict[qword]: kg_score[i][j] += cached_dict[qword][kword]  
                    j += 1
                #print(kg_score, kword, cached_dict)
        #print(i, query_map[i])
        flag = query_map[i]
        while(i < len(query_map) and flag == query_map[i]):
            try:
                kg_score[i] = kg_score[flag]
            except(IndexError):
                print(i, flag, kg_score.shape)
            i += 1 
    kg_score1 = np.concatenate((np.zeros((kg_score.shape[0], 1)), kg_score), 1)
    kg_score2 = np.concatenate((np.zeros((1, kg_score1.shape[1])), kg_score1), 0)
    return kg_score2

def word_segment_map(sent_list):
    ret_sent_list = []
    word_map_list = []
    for i in range(len(sent_list)):
        ret_sent_list.append([])
        word_map_list.append([])

    i = 0
    for sent in sent_list:
        cur_word = ""
        cur_punctuation = ""        
        for word in sent:
            
            word = word.lower()
            #import pdb; pdb.set_trace()
            if word[0] == '▁':
                if cur_punctuation != "" and cur_punctuation != cur_word: 
                    ret_sent_list[i][-1] = ret_sent_list[i][-1][:-1]
                    ret_sent_list[i].append(cur_punctuation)
                    word_map_list[i][-1] += 1
                cur_word = word[1:]
                cur_punctuation = ""
                word_map_list[i].append(word_map_list[i][-1] + 1 if len(word_map_list[i]) else 0)
                ret_sent_list[i].append(cur_word)
            elif word.isalnum():
                word_map_list[i].append(word_map_list[i][-1] if len(word_map_list[i]) else 0)
                ret_sent_list[i][-1] += word
                cur_word += word
                cur_punctuation = ""
            else:
                #import pdb;pdb.set_trace()
                if cur_punctuation == "":
                    #print("now1 ", word)
                    cur_punctuation += word
                    cur_word += word
                    word_map_list[i].append(word_map_list[i][-1] if len(word_map_list[i]) else 0)
                    ret_sent_list[i][-1] += word
                elif cur_punctuation == cur_word:
                    cur_punctuation += word
                    cur_word += word  
                    ret_sent_list[i].append(word)
                    word_map_list[i].append(word_map_list[i][-1] + 1 if len(word_map_list[i]) else 0)                                      
                else:        
                    #print("now2 ", word, cur_punctuation, word)            
                    ret_sent_list[i][-1] = ret_sent_list[i][-1][:-1]
                    ret_sent_list[i].append(cur_punctuation)
                    ret_sent_list[i].append(word)
                    word_map_list[i][-1] += 1
                    word_map_list[i].append(word_map_list[i][-1] + 1 if len(word_map_list[i]) else 0)
                    cur_word = word
                    cur_punctuation = word
        #print("test\n", cur_punctuation, cur_word)
        if cur_punctuation != "" and cur_punctuation != cur_word:
            ret_sent_list[i][-1] = ret_sent_list[i][-1][:-1]
            ret_sent_list[i].append(cur_punctuation)
            word_map_list[i][-1] += 1
        i += 1
    return ret_sent_list, word_map_list

def cal_reliability_tensor(word_map_list, ret_sent_list, content_lengths, net_dict, cached_dict, mem_len, content):
    #print("word len", len(word_map_list))
    #import pdb; pdb.set_trace()
    reliability_tensor = [] # list, (num of u, Bsz, qlen, klen)
    for i in range(len(word_map_list)):
        max_len = max(l for l in content_lengths[i])
        #print(content_lengths[i], len(word_map_list[i][0]), "sdad")
        tensor_for_timestep = []
        for bnum in range(len(word_map_list[i])):
            
            mat_for_bnum = np.zeros((content_lengths[i][bnum], 0))
            left_padding = 0
            right_padding = max_len - content_lengths[i][bnum]
            for j in range(i+1):
                kg_score = point2point_reliability(net_dict, cached_dict, ret_sent_list[i][bnum], ret_sent_list[j][bnum], word_map_list[i][bnum], word_map_list[j][bnum], content[i][bnum], content[j][bnum]) #to be done: 未设置步数
                
                if j >= i:
                    if mat_for_bnum.shape[1] > mem_len:
                        tmp_mat = np.transpose(mat_for_bnum)[-mem_len:]
                        mat_for_bnum = np.transpose(tmp_mat)
                    elif i > 0:
                        left_padding = mem_len - mat_for_bnum.shape[1]               
                mat_for_bnum = np.concatenate((mat_for_bnum, kg_score), 1) 
            
            mat_for_bnum = torch.tensor(mat_for_bnum, dtype=torch.float32)
            mat_for_bnum = torch.softmax(mat_for_bnum, 1)
            #print(mat_for_bnum.size())
            mat_for_bnum = torch.cat((torch.zeros(content_lengths[i][bnum], left_padding), mat_for_bnum, torch.zeros(content_lengths[i][bnum], right_padding)), 1)
            #print(mat_for_bnum.size())
            down_padding = mem_len + max_len if i else max_len
            mat_for_bnum = torch.cat((mat_for_bnum, torch.zeros(right_padding, down_padding)), 0)
            #print(mat_for_bnum.size())

            tensor_for_timestep.append(mat_for_bnum.tolist())
        reliability_tensor.append(torch.tensor(tensor_for_timestep, dtype=torch.float32))     
        #print("nn",torch.tensor(tensor_for_timestep  ).size()    ) 
    #print("end, ", torch.tensor(reliability_tensor).size()) 
     
    return reliability_tensor

#for debugging
def mmm(content, ret_sent_list, word_map_list):
    for i in range(len(word_map_list)):
        print(len(word_map_list[i]), len(ret_sent_list[i]), len(content[i]))
        import pdb; pdb.set_trace()
        for j in range(len(word_map_list[i])):
            print(content[i][j], ":", ret_sent_list[i][word_map_list[i][j]])

if __name__ == "__main__":
    cad = {}
    csv_path = "conceptnet-5.7.0-rel.csv"
    #dic = conceptnet_to_dict(csv_path)
    dic = {'how':{'are':1,'you':1},'are':{'you':1}}
    cal_path_reliability(dic, cad, 'how')

    print(cad)
    #print(di
'''
词语分配给自己的权重过小：用对角矩阵初始化？
'''