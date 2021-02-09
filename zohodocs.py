"""from pyzohodocs.pyzohodoc import ZohoDocsClient
client = ZohoDocsClient('3a9f41b774a563d07d3b04571c22db0f')
client.upload_file(file_name='rc_launcher.log',file_path = '/home/reorder/Documents/F3Nett/F3Net/rc_launcher.log')"""


from pyzohodocs.pyzohodoc import ZohoDocsClient
client = ZohoDocsClient('3a9f41b774a563d07d3b04571c22db0f')
client.create_file(file_name ='test.docx', service='document',type='doc')

#from pyzohodocs.pyzohodoc import ZohoDocsClient
#client = ZohoDocsClient('3a9f41b774a563d07d3b04571c22db0f')
#client.upload_file(file_name='_',file_path = '/home/reorder/model-175_v2_2')


curl https://apidocs.zoho.com/files/v1/upload?authtoken=3aXXXXXX\&scope=docsapi --request POST --data '{"filename":"1021_16_log.tar","content":`cat 1021_16_log.tar`}'
{"ERROR":"Invalid URL specified !!"}


