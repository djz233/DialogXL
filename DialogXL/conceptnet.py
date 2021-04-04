import pandas as pd
import numpy as np
#file = "conceptnet-5.7.0-processed_EN.csv"
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
    init_step = step
    query = set([key])
    bfs_next = set()
    while step: 
        for q in query:  
            distr = 0 
            next_step = set() 
                        
            for word, paths in net_dict[q].items():
                if not allow_loop and word in flag_set : continue
                else: 
                    distr += paths
                    next_step.add(word)

            resource = init_resource if step == init_step else cached_dict[key][q]
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
            


def point2point_reliability(net_dict, cached_dict, raw_text, steps=2, allow_loop=True):
    # text (B, text)
    max_len = 0
    text = []
    for str in raw_text:
        #print(str.split())
        text.append(str.lower().split())
        max_len = max(max_len, len(str.split()))
    #print(text[-1])
    kg_score = np.zeros((len(text), max_len, max_len))
    i,j,k = 0,0,0
    for str in text:
        j = 0
        for key in str:
            k = 0
            if key in net_dict:
                if key not in cached_dict:
                    cal_path_reliability(net_dict, cached_dict, key, step=steps, allow_loop=allow_loop)
                    #input()
                for query in str:  
                    kg_score[i][j][k] += (cached_dict[key][query] if query in cached_dict[key] else 0) 
                    k += 1   
                #print(kg_score, key, cached_dict)
            j += 1
        i += 1        
    return kg_score

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
                if cur_punctuation == "":
                    cur_punctuation += word
                    cur_word += word
                    word_map_list[i].append(word_map_list[i][-1] if len(word_map_list[i]) else 0)
                    ret_sent_list[i][-1] += word
                else:                    
                    ret_sent_list[i][-1] = ret_sent_list[i][-1][:-1]
                    ret_sent_list[i].append(cur_punctuation)
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

def cal_reliability_tensor(word_map_list, ret_sent_list, content_mask):
    reliability_tensor = [] # list, (Bsz, num of u, qlen, klen)
    for i in range(len(content_mask)):
        for bnum in range(len(content_mask[i])):
            for j in range(len(content_mask[i][bnum])):
                for 
    return reliability_tensor

#for debugging
def mmm(content, ret_sent_list, word_map_list):
    for i in range(len(word_map_list)):
        print(len(word_map_list[i]), len(ret_sent_list[i]), len(content[i]))
        import pdb; pdb.set_trace()
        for j in range(len(word_map_list[i])):
            print(content[i][j], ":", ret_sent_list[i][word_map_list[i][j]])

if __name__ == "__main__":
    dic = {'how': {'are':1,'you':1}, 'are': {'how':1, 'fine':1}, 'you':{'how':1, 'fine':1, 'thank':1}, 'fine':{'are':1, 'you':1}, 'thank':{'you':1}}
    cad = {}
    str = [" How are you Fine thank "]
    print(str)
    print(point2point_reliability(dic, cad, str, allow_loop=False))
    #print(cad)
    #print(dic)
'''
词语分配给自己的权重过小：用对角矩阵初始化？
'''