import sys
import os
import multitimer
import subprocess
import atexit

def executeC():
    completed = subprocess.run(['python3', 'publish_cart_id.py'])
    print('returncode:', completed.returncode)

"""def monitor():
    print("am checking")"""

def monitorr():
    print("sub_Process Monitor started at every 15 secs")
    for i in range(0,(len(line)-1)):
        program_name= (line[i].split(' '))[0]
        pid=os.popen("ps -ef | grep {0} | grep -v grep".format(program_name)).read()
        print(pid)
        if pid == '':
           print("CrashedError:",program_name,"is crashed :Restarting it again")
           os.system("python3 {0} ".format(line[i]))

def main():
    print("\n"+"sub-programs start executing")
    for i in range(0, (len(line)-1)):
        #print(line[i])
        program_name= (line[i].split(' '))[0]
        if not os.path.isfile(program_name):
                #print(program_name)
                print ("FileError:"+program_name,"is missing")
                sys.exit(1)
        os.system("python3 {0} ".format(line[i]))
        pid=os.popen("ps -ef | grep {0} | grep -v grep".format(program_name)).read()
        print(program_name,"is started executing "+"\n"+pid)

    #buffer_request_timer=multitimer.RepeatingTimer(interval=2, function=monitor).start()
    buffer_request_timer=multitimer.RepeatingTimer(interval=15, function=monitorr).start()
def close():
    os.system("killall python3")
    #for i in range(0, (len(line)-1)):
        #program_name= (line[i].split(' '))[0]
        #os.system("pkill {0} ".format(program_name))
    print("All python3 process killed")     

if __name__=='__main__':
    print("ID of main process: {}".format(os.getpid())) 
    command = sys.argv[1]
    config_txt = open("config.txt","r")
    #prg = config_txt.readlines()
    prg = config_txt.read()
    line= prg.split('\n')

    if command == 'test':
       for i in range(0, (len(line)-1)):
           program_name= (line[i].split(' '))[0]
           pid=os.popen("ps -ef | grep {0} | grep -v grep".format(program_name)).read()
           if pid != '':
              print("child process",program_name,"Kill it first")
       print("\n"+"No child_process is running")
       main()
       atexit.register(close)


    


