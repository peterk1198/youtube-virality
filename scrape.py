import os
import csv
import json

def cleanData(json_file_name, csv_file_name):
	print('fuck')
	json_file = open(json_file_name)
	json_str = json_file.read()
	json_dict = json.loads(json_str)
	cat_dict = {}
	for cat_ob in json_dict['items']:
		cat_dict[int(cat_ob['id'])] = cat_ob['snippet']['title']
	f = open(csv_file_name)
	csv_f = csv.reader(f)
	orig_vids = set()
	total_dup = 0
	tot_rows = 0
	
	with open(csv_file_name[0:2] + '_new.csv', 'a') as csvFile:
		writer = csv.writer(csvFile)
		# loop over each row in the csv
		for row in csv_f:
			# if we are at the first row then we change from category_id to anme
			if row[4] == 'category_id':
				row[4] == 'category_name'
				writer.writerow(row)
			else:
				tot_rows += 1
				# remove duplicate values from the csv
				if row[0] not in orig_vids:
					orig_vids.add(row[0])
					if int(row[4]) not in cat_dict:
						continue
					category = cat_dict[int(row[4])]
					row[4] = category
					writer.writerow(row)

if __name__ == '__main__':
    cleanData('CA_category_id.json', 'CAvideos.csv')
    