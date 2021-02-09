""""from log.create import Make_tar

Make_tar(100)"""
import os
os.system("curl -X POST 'https://docs.zoho.com/files/v1/upload?authtoken=3a9f41b774a563d07d3b04571c22db0f&filename=100_avi.tar' -H 'cache-control:no-cache' -F content=@/home/reorder/Logs/100_avi.tar")
print("Done")
