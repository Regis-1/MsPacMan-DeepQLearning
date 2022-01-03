import sys
import os
import re

pattern_train = '.+\[train\].+step (\d+) game (\d+\/\d+) avg loss: (\d\.?\d*) per_ab: (\d+\.?\d*\/\d+\.?\d*).+loss: (\d+\.?\d*\/\d+\.?\d*\/\d+\.?\d*).+ght: (\d+\.?\d*).+tal: (\d+\.?\d*) mem: (\d+).+fr: (\d+\.?\d*) eps: (\d+\.?\d*).+'
pattern_play = '.+\[play\] game (\d+) len: (\d+) score: (\d+\.?\d*)\/(\d+\.?\d*)\/(\d+\.?\d*)\/(\d+\.?\d*) fr: (\d+\.?\d*).*'

PLAY_FILE = 'play_data.csv'
TRAIN_FILE = 'train_data.csv'
PLAY_HEADER = 'game;len;score1;score2;score3;score4;fr\n'
TRAIN_HEADER = 'step;game1;game2;avg_loss;per_ab1;per_ab2;avg_l;min_l;max_l;max_weight;sum_total;memory;fr;epsilon\n'

logs=[]
trains_data = []
plays_data=[]

for (dirpath, dirnames, filenames) in os.walk('.'):
    logs.extend(filenames)
    break
    
logs = [f for f in logs if f != 'log2csv.py']

t_line_cnt=0
p_line_cnt=0

for log in logs:
    with open(log, 'r') as ofile:
        for line in ofile.readlines():
            match = re.search(pattern_train, line)
            if match:
                dict_train = {}
                dict_train['step']=match.group(1)
                dict_train['game']=match.group(2)
                dict_train['avg_loss']=match.group(3)
                dict_train['per_ab']=match.group(4)
                dict_train['avg_min_max_loss']=match.group(5)
                dict_train['max_weight']=match.group(6)
                dict_train['sum_total']=match.group(7)
                dict_train['memory']=match.group(8)
                dict_train['fr']=match.group(9)
                dict_train['epsilon']=match.group(10)
                trains_data.append(dict_train)
                t_line_cnt += 1
            else:
                match = re.search(pattern_play, line)
                if match:
                    dict_play = {}
                    dict_play['game']=match.group(1)
                    dict_play['len']=match.group(2)
                    dict_play['score1']=match.group(3)
                    dict_play['score2']=match.group(4)
                    dict_play['score3']=match.group(5)
                    dict_play['score4']=match.group(6)
                    dict_play['fr']=match.group(7)
                    plays_data.append(dict_play)
                    p_line_cnt += 1
                    
    print('File', log, 'finished.', 'T_lines:',t_line_cnt,'|','P_lines:',p_line_cnt)
    t_line_cnt=0
    p_line_cnt=0

with open(TRAIN_FILE, 'w') as f:
    f.write(TRAIN_HEADER)
    for d in trains_data:
        games = d['game'].split('/')
        per_abs = d['per_ab'].split('/')
        losses = d['avg_min_max_loss'].split('/')
        
        line = d['step']+';'+games[0]+';'+games[1]+';'+d['avg_loss']+';'+per_abs[0]+';'+per_abs[1]+';'+\
                losses[0]+';'+losses[1]+';'+losses[2]+';'+d['max_weight']+';'+d['sum_total']+';'+\
                d['memory']+';'+d['fr']+';'+d['epsilon']+';'+'\n'
        f.write(line)

with open(PLAY_FILE, 'w') as f:
    f.write(PLAY_HEADER)
    for d in plays_data:
        line = d['game']+';'+d['len']+';'+d['score1']+';'+d['score2']+';'+d['score3']+';'+d['score4']+\
                ';'+d['fr']+'\n'
        f.write(line)

print('Done!')