import pandas as pd

# Function to format the row values with quotations
def add_quotes(row):
 values = row.split(" ")
 print(values)
 output = ["'{}'".format(value) for value in values if value.strip()]
 return ",".join(output)

# Function that converts csv string data into sql insert statements 
def csv_to_mysql(table, data):
 csv_array = data.split("\n")
 header_str = add_quotes(csv_array[0])
 sql_statement = ""
 for row in csv_array[1:-1]:
    row_str = add_quotes(row)
    sql_statement += "INSERT INTO {}({}) VALUES({}); \n".format(table, header_str, row_str)

 return sql_statement

# Function that will write the generated sql statement into txt file
def generate_insert_file(file_name, sql_statement):
   with open(file_name, "w") as f:
    f.write(sql_statement)


def main():
    csv_data = pd.read_csv('../data/ObesityDataSet_raw_and_data_sinthetic.csv')
    csv_string = csv_data.to_string(index=False)
    
    sql = csv_to_mysql('Obesity',csv_string)
    generate_insert_file("insert_statement.txt", sql)
    print("Execution Finished!")


if __name__ == "__main__":
    main()
