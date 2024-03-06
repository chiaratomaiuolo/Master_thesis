__description__= \
"""This script takes as input pathfiles to .txt files and merge them without
linespaces between rows of diffierent files. 
It returns a unique file containing the merge of all the others. 
"""
def merge_files(input_files, output_file):
    with open(output_file, 'w') as out_file:
        for file_path in input_files:
            with open(file_path, 'r') as in_file:
                out_file.write(in_file.read())

if __name__ == "__main__":
    #Data from 12/2 to 22/2 
    '''
    input_files = ['/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/bfslog_2024-02-12_1226.txt',\
                     '/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/bfslog_2024-02-13_0000.txt',\
                     '/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/bfslog_2024-02-14_0000.txt',\
                     '/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/bfslog_2024-02-15_0000.txt',\
                     '/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/bfslog_2024-02-15_1525.txt',\
                     '/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/bfslog_2024-02-15_1546.txt',\
                     '/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/bfslog_2024-02-16_0000.txt',\
                     '/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/bfslog_2024-02-16_1442.txt',\
                     '/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/bfslog_2024-02-16_1453.txt',\
                     '/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/bfslog_2024-02-16_1454.txt',\
                     '/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/bfslog_2024-02-16_1500.txt',\
                     '/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/bfslog_2024-02-16_1807.txt',\
                     '/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/bfslog_2024-02-16_1811.txt',\
                     '/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/bfslog_2024-02-17_0000.txt',\
                     '/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/bfslog_2024-02-18_0000.txt',\
                     '/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/bfslog_2024-02-19_0000.txt',\
                     '/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/bfslog_2024-02-19_1106.txt',\
                     '/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/bfslog_2024-02-19_1550.txt',\
                     '/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/bfslog_2024-02-20_0000.txt',\
                     '/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/bfslog_2024-02-20_1238.txt',\
                     '/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/bfslog_2024-02-20_1859.txt',\
                     '/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/bfslog_2024-02-21_0000.txt',\
                     '/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/bfslog_2024-02-22_0000.txt']
    output_file = "/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/merged_DME_measurements.txt"
    '''
    #AC containing epoxy samples DME filled - from 26/2 at ~16:00
    input_files = ["/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/bfslog_2024-02-26_0000.txt",\
                   "/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/bfslog_2024-02-27_0000.txt",\
                   "/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/bfslog_2024-02-27_0933.txt",\
                   "/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/bfslog_2024-02-28_0000.txt",\
                   "/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/bfslog_2024-02-29_0000.txt",\
                   "/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/bfslog_2024-03-01_0000.txt",\
                   "/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/bfslog_2024-03-02_0000.txt",\
                   "/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/bfslog_2024-03-03_0000.txt",\
                   "/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/bfslog_2024-03-04_0000.txt",\
                   "/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/bfslog_2024-03-05_0000.txt"]
    output_file = "/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/merged_measurements_DME_with_epoxysamples.txt"

    merge_files(input_files, output_file)
    print("Merge Completed. Output file is:", output_file)