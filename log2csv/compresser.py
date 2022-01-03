import sys

PREFIX = 'conv_'

def main():
    if len(sys.argv) != 3:
        print('Wrong number of arguments!')
        sys.exit()
    
    offset = int(sys.argv[2])
    cnt = 0
    write_cnt = 0
    
    filename = sys.argv[1]
    with open(filename, 'r') as i_f:
        with open(PREFIX+filename, 'w') as o_f:
            for line in i_f.readlines():
                if (cnt % offset) == 0: 
                    o_f.write(line)
                    write_cnt += 1
                    
                cnt += 1
    
    print('Finished')
    print('All lines:', cnt, '||', 'Rewritten lines:', write_cnt)
    
if __name__ == '__main__':
    main()