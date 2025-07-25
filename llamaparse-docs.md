### sample code

```
from llama_cloud_services import LlamaParse

parser = LlamaParse(
  # See how to get your API key at https://docs.cloud.llamaindex.ai/api_key
  api_key="<you-api-key>",
)



# Example usage:

# sync
result = parser.parse("./my_file.pdf")

# sync batch
results = parser.parse(["./my_file1.pdf", "./my_file2.pdf"])

# async
result = await parser.aparse("./my_file.pdf")

# async batch
results = await parser.aparse(["./my_file1.pdf", "./my_file2.pdf"])

# get the llama-index markdown documents
markdown_documents = result.get_markdown_documents(split_by_page=True)

# get the llama-index text documents
text_documents = result.get_text_documents(split_by_page=False)

# get the image documents
image_documents = result.get_image_documents(
    include_screenshot_images=True,
    include_object_images=False,
    # Optional: download the images to a directory
    # (default is to return the image bytes in ImageDocument objects)
    image_download_dir="./images",
)

# access the raw job result
# Items will vary based on the parser configuration
for page in result.pages:
    print(page.text)
    print(page.md)
    print(page.images)
    print(page.layout)
    print(page.structuredData)
```