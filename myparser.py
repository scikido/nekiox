import time
from llmware.library import Library
from llmware.retrieval import Query
import re
import json

input_fp = r"C:\Users\DELL\Downloads\Alternative Final Project\docs"
output_fp = r"C:\Users\DELL\Downloads\Alternative Final Project\outputs"

def parsing_with_docs():
    t0 = time.time()

    lib = Library().create_new_library("new_pdf")

    parsing_output = lib.add_files(input_folder_path=input_fp)

    print("update: parsing time - ", time.time() - t0)
    print("update: parsing_output - ", parsing_output)

    output1 = lib.export_library_to_jsonl_file(output_fp, "new_output")

    output2 = Query(lib).export_all_tables(query="",output_fp=output_fp)

    return 0

def getting_text(path):
    with open(path,'r') as file:
        text = r"" + file.read()
        pattern = "{.*}"
        matches = re.findall(pattern, text)
        store = ""
        for match in matches:
            file = json.loads(match, strict=False)
            print(file['text_search'])
            store += file['text_search']
        return store
    

# q = parsing_with_docs()
# p = parse_using_llama(r"C:\Users\DELL\Downloads\table extract\pdf_tables\new_output.jsonl")
# print(p)