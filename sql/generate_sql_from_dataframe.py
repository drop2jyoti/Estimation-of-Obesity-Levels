import pandas as pd
import sqlalchemy as sql
#GENERATE SQL CREATE STATEMENT FROM DATAFRAME
#SOURCE = df
#TARGET = table_name
#GENERATE SQL CREATE STATEMENT FROM DATAFRAME

def SQL_CREATE_STATEMENT_FROM_DATAFRAME(SOURCE, TARGET):

    # SQL_CREATE_STATEMENT_FROM_DATAFRAME(SOURCE, TARGET)
    # SOURCE: source dataframe
    # TARGET: target table to be created in database
    
    sql_text = pd.io.sql.get_schema(SOURCE.reset_index(), TARGET)  
    #Check the SQL CREATE TABLE Statement String
    print(sql_text)
    return sql_text


#GENERATE SQL INSERT STATEMENT FROM DATAFRAME
def SQL_INSERT_STATEMENT_FROM_DATAFRAME(SOURCE, TARGET):
    sql_texts = []
    for index, row in SOURCE.iterrows():       
        sql_texts.append('INSERT INTO '+TARGET+' ('+ str(', '.join(SOURCE.columns))+ ') VALUES '+ str(tuple(row.values)))   
    #Check the SQL INSERT INTO Statement String
    print('\n\n'.join(sql_texts))     
    return sql_texts
    

def main():
    csv_data = pd.read_csv('../data/ObesityDataSet_raw_and_data_sinthetic.csv')
    print(csv_data)
    create_table = SQL_CREATE_STATEMENT_FROM_DATAFRAME(csv_data, "obesity")
    
if __name__ == "__main__":
    main()