import tarfile
import pysftp
import sys
import os

def Make_tar(Filename):
    Tar_file=str(Filename)+".tar"
    os.chdir("/home/reorder/")
    print(os.getcwd())
    os.system("tar -cvf {0} Logs/* sequences/*".format(Tar_file))
    #tar -cvf Mytar.tar Logs/* sequences/*
    print("waiting For zoho response:")
    command = "curl -X POST 'https://docs.zoho.com/files/v1/upload?authtoken=3a9f41b774a563d07d3b04571c22db0f&filename="+str(Filename)+".tar&fid=kj174130450c4dd684d909117fbab4a0b19f4' "+"-H 'cache-control:no-cache' -F content=@{0}"  
    print(command)
    #os.system(command.format(Tar_file))
    print("Zoho_docs Upload completed.")
    with pysftp.Connection('192.168.0.21', username='econ-reorder', password='reorder') as sftp:
        with sftp.cd('/home/econ-reorder/'):
            sftp.put(Tar_file)
    #command_sftp = "sftp econ-reorder@192.168.0.21"
    #os.system(command_sftp)
    #put_command = "put Tar_file /home/econ-reorder/"
    os.system("rm -rf /home/reorder/Logs/*")
    os.system("rm -rf /home/reorder/sequences/*")
    print("SFTP Upload completed.")
#    os.system("rm /home/reorder/{0}.tar".format(Filename))

        
if __name__=='__main__':      
        tar_filename = sys.argv[1]
        Make_tar(tar_filename)
        
        


