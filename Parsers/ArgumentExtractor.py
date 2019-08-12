import os
while True:
    #this file is for parsing files from the BECAUSE ANN corpus
    fname = input('Enter filename:')
    if fname == 'stop':
        break
    x = fname + '.ann'
    if os.path.exists(x):
        with open(x,'r',encoding='utf8') as ifile:
            #writefile
            ofile = open('pairs.txt','a',encoding='utf8')    
            #create a dictionary of the arguments in each document
            args = dict()  
            for line in ifile:
                ##print(line.strip())
                entry = line.strip().split('\t')
                ##print(entry[0])
                if entry[0].find('T') != -1: #if article text type
                    args[entry[0]] = entry[2]
            ifile.seek(0)
            ##print(args)
            for line in ifile:
                entry = line.strip().split('\t')
                if entry[0].find('E') != -1: #if relation type
                    ##print(entry[0])
                    parts = entry[1].split(' ')
                    ##print(parts)
                    if len(parts) == 3 and (parts[0].find('Consequence') != -1 or parts[0].find('Motivation') !=-1 or parts[0].find('Purpose')!=-1):
                        if parts[1].find('Cause') != -1:
                            cause = parts[1].split(':')
                            effect = parts[2].split(':')
                            print(args[cause[1]],'\t',args[effect[1]])
                            ofile.write('{}\t{}\n'.format(args[cause[1]],args[effect[1]]))
                        else:
                            cause = parts[2].split(':')
                            effect = parts[1].split(':')
                            print(args[cause[1]],'\t',args[effect[1]])
                            ofile.write('{}\t{}\n'.format(args[cause[1]],args[effect[1]]))
                            
            ofile.close()
    else:
        print('file does not exist, try another')
