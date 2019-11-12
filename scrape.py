import os
import csv
import json
import heapq
def cleanData(json_file_name, csv_file_name):
	# gets the number associated with eah category name
	cat_dict = mapCats(json_file_name)
	cat_virality = initClean(csv_file_name, cat_dict)
	fin_clean(csv_file_name, cat_virality)


# returns a file reader given a specific filename
#input the file name as a string
def mapCats(json_file_name):
	json_file = open(json_file_name)
	json_str = json_file.read()
	json_dict = json.loads(json_str)
	cat_dict = {}
	for cat_ob in json_dict['items']:
		cat_dict[int(cat_ob['id'])] = cat_ob['snippet']['title']
	return cat_dict


# does the inital scrape of our cleaning
# - removes duplicates for the file
# - adds the category name as a column in our csv
# -returns a dictionary a dictionary of dictionaries where each key is a category and each value is a 
#  containing the key, values of video id and views

def initClean(csv_file_name, cat_dict):
		f = open(csv_file_name)
		csv_f = csv.reader(f)
		orig_vids = set()
		total_dup = 0
		tot_rows = 0
		cat_virality = {}
		# remove duplicates and add the category names to the file
		csvFile = open(csv_file_name[0:2] + 'init_clean.csv', 'w')
		writer = csv.writer(csvFile)
		# loop over each row in the csv
		for row in csv_f:
			# if we are at the first row then we change from category_id to anme
			if row[4] == 'category_id':
				row[4] = 'category_name'
				print(row)
				writer.writerow(row)
			else:
				tot_rows += 1
				# remove duplicate values from the csv
				if row[0] not in orig_vids:
					orig_vids.add(row[0])
					if int(row[4]) not in cat_dict:
						continue
					category = cat_dict[int(row[4])]
					if category not in cat_virality:
						cat_virality[category] = {row[0]: int(row[7])}
					else:
						cat_virality[category][row[0]] = int(row[7])
					row[4] = category
					writer.writerow(row)
		csvFile.close()
		return(cat_virality)

# gets te top percent of dictionary keys gicen a particular percentage
# percent is a integer value from 0 to 100
# vir_dict is a dict of videos from a given category

def getTopPercent(percent, vir_dict):
	n = round(len(vir_dict) * (percent/100))
	top_vid_ids = heapq.nlargest(n, vir_dict.keys(), key=lambda k: vir_dict[k])
	return top_vid_ids

def fin_clean(csv_file_name, cat_virality):
	viral_vids = set()
	for key,val in cat_virality.items():
		viral_vids = set(list(viral_vids) + getTopPercent(10, val))
	
	f = open(csv_file_name[0:2] + 'init_clean.csv')
	csv_f = csv.reader(f)
	
	csvFile =  open(csv_file_name[0:2] + 'virality.csv', 'w')
	writer = csv.writer(csvFile)
	# loop over each row in the csv
	n_rows = 0

	for row in csv_f:
		print(n_rows)
		# if we are at the first row then we change from category_id to anme
		if n_rows == 0:
			row.append('viral')
			writer.writerow(row)
		else:
			# remove duplicate values from the csv
			category = row[4]
			if category in cat_virality:
				if len(cat_virality[category]) < 100:
					continue
				if row[0] in viral_vids:
					row.append(1)
				else:
					row.append(0)
				writer.writerow(row)
		n_rows += 1
	csvFile.close()

if __name__ == '__main__':
    cleanData('CA_category_id.json', 'CAvideos.csv')
    