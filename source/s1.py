import datetime
print("this is 1")
print(datetime.datetime.now())
f = open('time/output.txt', 'a')
f.write("Started at \n")
f.write(str(datetime.datetime.now())+'\n')
f.write("Ended at \n")
f.write(str(datetime.datetime.now())+'\n')
f.write("This is time \n")
f.close()

# file_path = '/Users/bar.txt' 
# print os.path.dirname(file_path)  
# # /Users/Foo/Desktop

# if not os.path.exists(os.path.dirname(file_path)):
#     os.mkdirs(os.path.dirname(file_path))  
#     # recursively create directories if necessary

# with open(file_path, "a") as my_file:
#     # mode a will either create the file if it does not exist
#     # or append the content to its end if it exists.
#     my_file.write("your_text_to_append")
