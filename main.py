from llama_parse import LlamaParse
import time

def parse_using_llama(pdf):
    parser = LlamaParse(
        api_key="llx-D9kZUjUTnRxsPFwtmN50di1cpBdZcVLWoYP9DBegy0JAX7aG",
        result_type="markdown",
        verbose=True,
        language="en",
        num_workers=2,
    )

    documents = parser.load_data(pdf)
    return documents[0].text

def main():
    start_time = time.time()  # Start measuring time

    docs_text = parse_using_llama(r"po_test.pdf")
    print(docs_text)

    end_time = time.time()  # Stop measuring time
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

# Invoking the main function
if __name__ == "__main__":
    main()
