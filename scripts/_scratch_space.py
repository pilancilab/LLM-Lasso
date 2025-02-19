# Example usage
if __name__ == "__main__":
    input_str = "Acute myocardial infarction (AMI)  and diffuse large B-cell lymphoma (DLBCL)"
    gene = ["AASS", "CLEC4D"]
    # result1, result2 = parse_category_strings(input_str)
    # print(result1, result2)
    print(pubmed_retrieval(gene, input_str, "gpt-4o"))

