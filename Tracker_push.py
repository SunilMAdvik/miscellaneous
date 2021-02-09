import os
import sys
import json

def read_main(Json):
	DataInside =0
	klist=[]
	for key in Json.keys():
		klist.append(key)
	print("Deffirent Keys avaliable:")
	print(klist)
	user_ModelType = input('\n'+"Select Key: ")
	for key in Json.keys():
		if key == user_ModelType:
			DataInside = Json[key]
			break
	if DataInside == 0:
		print("Selected key is Invalid")
		sys.exit()
	return DataInside

def Read_input(Idata):
	Datainside =0
	iklist=[]
	for i in Idata:
		for ikey in i.keys():
			iklist.append(ikey)
	print('\n'+"Deffirent Keys avaliable:")
	print(iklist)
	user_ModelType = input('\n'+"Select Key: ")
	for j in Idata:
		for key, value in j.items():
			if key == user_ModelType:
				Datainside = j[key]
				break
	if Datainside == 0:
		print("Selected key is Invalid")
		sys.exit()
	return Datainside

def takeData(iConfig):
	Temp = {}
	for confg in iConfig:
		Input = input("Give Full Path for ("+confg +") :")
		while Input == '':
			print("InputError:Entered Invalid Path, Give correct Path :" )
			Input = input("Give Full Path for ("+confg +") :")
		Temp[confg] = Input
	return Temp

def write_json(data, filename='data.json'): 
    with open(filename,'w') as f: 
        json.dump(data, f, indent=4) 
	
def PushData() :
	print("Select where you need to insert... "+"\n")
	ModelType = read_main(json_file)
	ModelKinds = Read_input(ModelType)
	ModelInput_list = ModelKinds[0]
	if len(ModelInput_list)==0:
		print("JsonError : Json Is need update Properly ")
		sys.exit()
	InputConfig = ModelInput_list['input_cofig']
	ModelInput_list = ModelKinds[1]
	ModelInput = ModelInput_list['Model_input']
	#versionsData = Read_input(ModelInput)
	inklist=[]
	for i in ModelInput:
		for ikey in i.keys():
			inklist.append(ikey)
	print('\n'+"These are avaliable versions in this Model Type:")
	print(inklist)
	VersionName = input("\n"+"Type Name of Version, you need to insert :")
	while VersionName == '':
		print("\n"+"InputError:Entered Invalid Version Name, re-Enter again :" )
		VersionName = input("\n"+"Type Name of Version, you need to insert :")
	if len(ModelInput)==0:
		print("\n"+"List is Empty :You are entering First version")
		ModelInput_dict = ModelInput
		ModelInput_dict[VersionName] = []
	ModelInput_dict = ModelInput[0]
	ModelInput_dict[VersionName] = []
	saveData = ModelInput_dict[VersionName]
	UserInputData = takeData (InputConfig)
	saveData.append(UserInputData)
	write_json(json_file)
	print("Data Is Pushed to Json_File")
	
	"""while True:
	print("\n"+"You want query more.. ?")
	more = input("Type 'Y' or 'N' :")
	if more == 'Y':
		continue
	else :
		break
	break"""

def Display(idata):
	print("Model Data avaliable :"+"\n")
	for l in idata:
		for k, v in l.items():
			print (k ,":",v)
	
def Query():
	while True:
		print('\n'+"Select Which type of model you are looking for..."+"\n")
		ModelType = read_main(json_file)
		ModelKinds = Read_input(ModelType)
		ModelInput_list = ModelKinds[1]
		ModelInput = ModelInput_list['Model_input']
		versionsData = Read_input(ModelInput)
		Display(versionsData)
		print("\n"+"You want query more.. ?")
		more = input("Type 'Y' or 'N' :")
		if more == 'Y':
			continue
		else :
			break
def main():
	if len(sys.argv) < 2 or sys.argv[1] == 'query':
		Query()
	elif sys.argv[1] == 'insert':
		PushData()
	else:
		print("Passed Wrong agruments: only (pushdata) or (query) is valid")


if __name__=='__main__':
	f = open('data.json')
	json_file = json.load(f)
	main()
