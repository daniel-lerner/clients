# -*- coding: utf-8 -*-
"""
This library was created to summarize all the functions used to clean and merge
clients based on CPF, name and address columns in a dataframe. It also counts
the number of occurrences of each value in a column and performs the fuzzy
between two columns row by row. In addition, it is possible to validate the
merge through fuzzy score and comparisons between first, second, third and
last name.

In order to clean the columns for merging the dataframes in the best possible
way, it is recommended to follow a specific order of the functions. Some of
them are optional.

The order for CPFs follows:
    1) column_to_string
    2) delete_non_numeric
    3) add_zeros_to_cpf
    4) replace_zeroed_cpfs
    5) merge_dataframes
    6) validate_names
    7) split_validated_and_non
    8) fuzzy_row_between_two_columns
    9) validate_score
    
The order for names follows:
    1) column_to_string
    2) duplicate_column
    3) replace_accents
    4) lower_letters
    5) zeros_after_spaces (optional)
    6) zero_for_o (optional)   
    7) delete_numbers (optional)
    8) delete_duplicate_letters
    9) delete_non_alphabetic
    10) delete_double_spaces
    11) merge_dataframes

The order for address follows:
    1) column_to_string
    2) duplicate_column
    3) replace_accents
    4) lower_letters
    5) delete_duplicate_letters
    6) street_details  
    7) address_details
    8) delete_postcode
    9) delete_non_alphanumeric
    10) delete_double_spaces
    11) merge_dataframes

There are also some functions used to check columns data:
    1) list_of_chars
    2) filter_df_by_criterion
    3) filter_list_by_criterion

@author: dlerner
"""

from unidecode import unidecode
from fuzzywuzzy import fuzz
import re
import pandas as pd
import numpy as np
import pickle


#%% =============== BEGINNING ===============  


def column_to_string(df, column):
    """
    This function changes the type of the specified column to string. 

    Parameters
    ----------
    df : Dataframe
        Dataframe that will be changed.
    column : String
        Column that will be changed.

    Returns
    -------
    None.

    """
    df[column] = df[column].astype(str)


def duplicate_column(df, column, column_new):
        
    """
    This function duplicates the column that will be cleaned. The
    objective is to use the new column for modifications due to
    cleaning, mantaining the orginal without modifications.

    Parameters
    ----------
    df : Dataframe
        Dataframe that will be changed.
    column : String
        Original column.
    column_new : String
        New column.

    Returns
    -------
    None.

    """
    df[column_new] = df.loc[:,column]


def lower_letters(df, column):
    """
    This function changes all letters to lowercase in the
    specified column.

    Parameters
    ----------
    df : Dataframe
        Dataframe that will be changed.
    column : String
        Column that will be changed.

    Returns
    -------
    None.

    """
    df[column] = df[column].str.lower()

        
def replace_accents(df, column):
    """
    This function replaces the letters with accents to the
    sameones without accent in the specified column.

    Parameters
    ----------
    df : Dataframe
        Dataframe that will be changed.
    column : String
        Column that will be changed.

    Returns
    -------
    None.

    """
    df[column] = df[column].apply(unidecode)


#%% =============== CPF ONLY ===============


def add_zeros_to_cpf(df, column):
    """
    This fuction takes the values that have lenght lower then 11
    and add zeros until it get to 11 in the specified column.

    Parameters
    ----------
    df : Dataframe
        Dataframe that will be changed.
    column : String
        Column that will be changed.

    Returns
    -------
    None.

    """
    df[column] = df[column].apply(lambda x: x.zfill(11))


def delete_non_numeric(df, column):
    """
    This function deletes all non numeric chars in the specified
    column.

    Parameters
    ----------
    df : Dataframe
        Dataframe that will be changed.
    column : String
        Column that will be changed.

    Returns
    -------
    None.

    """
    df[column] = df[column].apply(lambda x: re.sub('[^0-9]', '', x))


def replace_zeroed_cpfs(df, column):
    """
    This function replaces zeroed cpfs for nan.

    Parameters
    ----------
    df : Dataframe
        Dataframe that will be changed.
    column : String
        Column that will be changed.

    Returns
    -------
    None.

    """
    df[column].replace('00000000000', np.nan, inplace=True)


#%% =============== NAME ONLY ===============        


def zeros_after_spaces(df, column):
    """
    This function deletes all zeros that come after a space.
    It is used only for name columns.

    Parameters
    ----------
    df : Dataframe
        Dataframe that will be changed.
    column : String
        Column that will be changed.

    Returns
    -------
    None.

    """
    df[column] = df[column].str.replace(' 0', ' ')



def zero_for_o(df, column):
    """
    This function replaces all zeros with the letter O.
    It is used only for name columns.

    Parameters
    ----------
    df : Dataframe
        Dataframe that will be changed.
    column : String
        Column that will be changed.

    Returns
    -------
    None.

    """  
    df[column] = df[column].str.replace('0', 'o')
    
    
def delete_numbers(df, column):
    """
    This function replaces all numbers for spacebar ' ' in the
    specified column.

    Parameters
    ----------
    df : Dataframe
        Dataframe that will be changed.
    column : String
        Column that will be changed.

    Returns
    -------
    None.

    """
    df[column] = df[column].apply(lambda x: re.sub('[0-9]', ' ', x))


def delete_non_alphabetic(df, column):
    """
    This function deletes all non alphabetic chars in the
    specified column.

    Parameters
    ----------
    df : Dataframe
        Dataframe that will be changed.
    column : String
        Column that will be changed.

    Returns
    -------
    None.

    """
    df[column] = df[column].apply(lambda x: re.sub('[^A-Za-z ]', '', x))


#%% =============== ADDRESS ONLY ===============      


def street_details(street):
    x = street
    if x[0:1] == 'r' and 'r.' in x or 'r:' in x or 'r;' in x:
        street = x.replace('r.', 'rua ').replace('r:', 'rua ').replace('r;', 'rua ')
    if x[0:2] == 'av' and 'av.' in x or 'av:' in x or 'av;' in x  or 'avn' in x:
        street = x.replace('av.', 'avenida ').replace('av:', 'avenida ').replace('av;', 'avenida ').replace('avn', 'avenida ')
    if 'est.' in x or 'est:' in x or 'est;' in x  or 'estr.' in x or 'estr:' in x or 'estr;' in x:
        street = x.replace('est.', 'estrada ').replace('est:', 'estrada ').replace('est;', 'estrada ').replace('estr.', 'estrada ').replace('estr:', 'estrada ').replace('estr;', 'estrada ')
    if 'rod.' in x or 'rod:' in x or 'rod;' in x  or 'gov.' in x or 'gov:' in x or 'gov;' in x:
        street = x.replace('rod.', 'rodovia ').replace('rod:', 'rodovia ').replace('rod;', 'rodovia ').replace('gov.', 'governador ').replace('gov:', 'governador ').replace('gov;', 'governador ')
    if 'prof.' in x or 'prof:' in x or 'prof;' in x  or 'cap.' in x or 'cap:' in x or 'cap;' in x:
        street = x.replace('prof.', 'professor ').replace('prof:', 'professor ').replace('prof;', 'professor ').replace('cap.', 'capitao ').replace('cap:', 'capitao ').replace('cap;', 'capitao ')
    if 'pres.' in x or 'pres:' in x or 'pres;' in x  or 'dr.' in x or 'dr:' in x or 'dr;' in x:
        street = x.replace('pres.', 'presidente ').replace('pres:', 'presidente ').replace('pres;', 'presidente ').replace('dr.', 'doutor ').replace('dr:', 'doutor ').replace('dr;', 'doutor ')
    return street


def apply_func_street(df, column):
    df[column] = df[column].apply(street_details)


def address_details(address):
    x = address
    if ' es,' in x  or ' es ' in x or '/es' in x or 'espirito santo' in x  or ' mg,' in x  or ' mg ' in x or '/mg' in x or 'minas gerais' in x:
        address = x.replace(' es,', ' ').replace(' es ', ' ').replace('/es', ' ').replace('espirito santo', ' ').replace(' mg,', ' ').replace(' mg ', ' ').replace('/mg', ' ').replace('minas gerais', ' ')
    if ' b.' in x or 'bairro' in x or ' ap:' in x or ' ap.' in x or 'apto' in x or 'apartamento' in x or '°' in x:
        address = x.replace(' b.', ' ').replace('bairro', ' ').replace(' ap:', ' ').replace(' ap.', ' ').replace('apto', ' ').replace('apartamento', ' ').replace('°', 'o')
    if ' n.' in x or ' n:' in x or ' n;' in x or ' n ' in x or ' no.' in x or ' no:' in x or ' no;' in x or ' no ' in x or ' numero' in x:
        address = x.replace(' n.', ' ').replace(' n:', ' ').replace(' n;', ' ').replace(' n ', ' ').replace(' no.', ' ').replace(' no:', ' ').replace(' no;', ' ').replace(' no ', ' ').replace(' numero', ' ')
    if ' n1' in x or ' n2' in x or ' n3' in x or ' n4' in x or ' n5' in x:
        address = x.replace(' n1', ' 1').replace(' n2', ' 2').replace(' n3', ' 3').replace(' n4', ' 4').replace(' n5', ' 5')
    if  ' n6' in x or ' n7' in x or ' n8' in x or ' n9' in x or ' n0' in x:
        address = x.replace(' n6', ' 6').replace(' n7', ' 7').replace(' n8', ' 8').replace(' n9', ' 9').replace(' n0', ' 0')
    if ' no1' in x or ' no2' in x or ' no3' in x or ' no4' in x or ' no5' in x:
        address = x.replace(' no1', ' 1').replace(' no2', ' 2').replace(' no3', ' 3').replace(' no4', ' 4').replace(' no5', ' 5')
    if ' no6' in x or ' no7' in x or ' no8' in x or ' no9' in x or ' no0' in x:
        address = x.replace(' no6', ' 6').replace(' no7', ' 7').replace(' no8', ' 8').replace(' no9', ' 9').replace(' no0', ' 0')
    if ' cep ' in x:
        address = x.replace('cep ', ' ')
    if x[0:3] == 'av ':
        address = x.replace('av ', 'avenida ')
    return address


def apply_func_address(df, column):
    df[column] = df[column].apply(address_details)
    
    
def delete_postcode(df, column):
    delete_double_spaces(df, column)
    df[column] = df[column].apply(lambda x: re.sub(r' \d{2} \d{5} \d{4} ', '', x))        #cellphone
    df[column] = df[column].apply(lambda x: re.sub(r' \d{2} \d{4} \d{5} ', '', x))        #cellphone
    df[column] = df[column].apply(lambda x: re.sub(r' \d{2} \d{1} \d{4} \d{4} ', '', x))  #cellphone
    df[column] = df[column].apply(lambda x: re.sub(r' \d{9} ', '', x))                    #cellphone
    df[column] = df[column].apply(lambda x: re.sub(r' \d{11} ', '', x))                   #cellphone
    df[column] = df[column].apply(lambda x: re.sub(r' \d{2} \d{4} \d{4} ', '', x))        #telephone
    df[column] = df[column].apply(lambda x: re.sub(r' \d{5} \d{3} ', '', x))
    df[column] = df[column].apply(lambda x: re.sub(r' \d{8} ', '', x))
    df[column] = df[column].apply(lambda x: re.sub(r' \d{2} \d{3} \d{3} ', '', x))
    df[column] = df[column].apply(lambda x: re.sub(r' \d{2}.\d{6} ', '', x))
    

def delete_non_alphanumeric(df, column):
    """
    This function replaces all alphanumeric chars for spacebar ' '
    in the specified column.

    Parameters
    ----------
    df : Dataframe
        Dataframe that will be changed.
    column : String
        Column that will be changed.

    Returns
    -------
    None.

    """
    df[column] = df[column].apply(lambda x: re.sub('[^A-Za-z0-9 ]', ' ', x))


#%% =============== FOR ALL ===============      


def delete_duplicate_letters(letters):
    x = letters
    if 'aa' in x or 'ee' in x or 'ii' in x or 'oo' in x or 'uu' in x:
        letters = x.replace('aa', 'a').replace('ee', 'e').replace('ii', 'i').replace('oo', 'o').replace('uu', 'u')
    if 'qq' in x or 'ww' in x or 'tt' in x or 'yy' in x or 'pp' in x:
        letters = x.replace('qq', 'q').replace('ww', 'w').replace('tt', 't').replace('yy', 'y').replace('pp', 'p')
    if 'dd' in x or 'ff' in x or 'gg' in x or 'hh' in x or 'jj' in x:
       letters = x.replace('dd', 'd').replace('ff', 'f').replace('gg', 'g').replace('hh', 'h').replace('jj', 'j')
    if 'kk' in x or 'll' in x or 'zz' in x or 'xx' in x or 'cc' in x:
        letters = x.replace('kk', 'k').replace('ll', 'l').replace('zz', 'z').replace('xx', 'x').replace('cc', 'c')
    if 'vv' in x or 'bb' in x or 'nn' in x or 'mm' in x:
        letters = x.replace('vv', 'v').replace('bb', 'b').replace('nn', 'n').replace('mm', 'm')
    return letters


def apply_func_letters(df, column):
    df[column] = df[column].apply(delete_duplicate_letters)
    
    
def delete_double_spaces(df, column):
    """
    This function replaces all double spaces for single spaces
    in the specified column.

    Parameters
    ----------
    ddf : Dataframe
        Dataframe that will be changed.
    column : String
        Column that will be changed.

    Returns
    -------
    None.

    """
    df[column] = df[column].apply(lambda x: re.sub(' +', ' ', x))  
    df[column] = df[column].str.lstrip()
    df[column] = df[column].str.rstrip()


#%% =============== MARK DUPLICATES =============== 


def mark_duplicates_cpf(df, column_index, column_cpf, column_name, column_address, column_new_index, column_merged_name, merge_type):
    column_index_x = column_index + '_x'
    column_index_y = column_index + '_y'
    column_name_x = column_name + '_x'
    column_name_y = column_name + '_y'
    column_address_x = column_address + '_x'
    column_address_y = column_address + '_y'
    
    # Discarding the values that should not be aconsidered as duplicate
    df_temp = df.dropna(subset=[column_cpf, column_name])
    df_temp = df_temp[(df_temp[column_cpf] != '00000000000') & (df_temp[column_name] != 'nan')]
    # Merging the dataframe with it self based on CPF
    merge_temp = merge_dataframes(df_temp, df_temp, column_cpf, column_cpf, 'CPF_merged')
    # Discading the exact match by index
    merge_temp = merge_temp[merge_temp[column_index_x] != merge_temp[column_index_y]]
    # Fuzzy score by row between the columns of name
    fuzzy_row_between_two_columns(merge_temp, column_name_x, column_name_y, 'fuzzy_score')
    # Separate dataframe between name score higher or equal then 80 and lower
    merge_80_higher_temp = merge_temp[(merge_temp['fuzzy_score'] >= 80) & (merge_temp[(column_name_x).split()[0]] == merge_temp[(column_name_y).split()[0]])]
    merge_80_lower_temp = merge_temp[(merge_temp['fuzzy_score'] < 80) | (merge_temp[(column_name_x).split()[0]] != merge_temp[(column_name_y).split()[0]])]
    # Creating columns for validation of lower scores
    validate_names(merge_80_lower_temp, column_name_x, column_name_y)
    # Spliting the dataframe between validated and not validated
    merge_validated, merge_not_validated = split_validated_and_non(merge_80_lower_temp)
    # Adding higher scores to final validated
    merge_validated = pd.concat([merge_validated, merge_80_higher_temp])
    
    # Create dictionay for the dups 
    dic_dup = dict()
    for index, row in merge_validated.iterrows():
        key = row[column_index_x]
        if not (key in dic_dup):
            dic_dup[key] = [row[column_index_y]]
        else:
            dic_dup[key].append(row[column_index_y])
    # Create new reference
    dic_new_ref = dict()
    for ref,dups in dic_dup.items():
        dups.append(ref)
        dups.sort()
        new_index = str(dups[0])
        for i in dups[1:]:
            new_index += '_' + str(i)
        dic_new_ref[ref] = new_index
    # Feed column_new_index
    df[column_new_index] = df[column_index].apply(lambda dups: dic_new_ref.get(dups,''))


def mark_duplicates_name(df, column_index, column_name, column_address, column_new_index):
    # Discarding the values that should not be aconsidered as duplicate
    df_temp = df.dropna(subset=[column_name, column_address])
    df_temp = df[(df[column_name] != 'nan') & (df[column_address] != 'nan')]
    # Merging the dataframe with it self based on name
    merge_temp = merge_dataframes(df_temp, df_temp, column_name, column_name, 'Name_merged')
    # Discading the exact match by index
    merge_temp = merge_temp[merge_temp[column_index + '_x'] != merge_temp[column_index + '_y']]
    # Validating the merge by fuzzy address
    fuzzy_row_between_two_columns(merge_temp, column_address + '_x', column_address + '_y', 'fuzzy_score')
    merge_validated = merge_temp[(merge_temp['fuzzy_score'] >= 80) & (merge_temp[(column_address + '_x').split()[0]] == merge_temp[(column_address + '_y').split()[0]])]
    # Create dictionay for the dups 
    dic_dup = dict()
    column_index_x = column_index + '_x'
    column_index_y = column_index + '_y'
    for index, row in merge_validated.iterrows():
        key = row[column_index_x]
        if not (key in dic_dup):
            dic_dup[key] = [row[column_index_y]]
        else:
            dic_dup[key].append(row[column_index_y])
    # Create new reference
    dic_new_ref = dict()
    for ref,dups in dic_dup.items():
        dups.append(ref)
        dups.sort()
        new_index = dups[0]
        for i in dups[1:]:
            new_index += '_' + i
        dic_new_ref[ref] = new_index
    # Feed column_new_index
    df[column_new_index] = df[column_index].apply(lambda dups: dic_new_ref.get(dups,''))


#%% =============== MERGE ===============  


def merge_dataframes(df1, df2, column_df1, column_df2, column_new, merge_type = 'inner'):
    
    """
    This function takes two columns of two dataframes and change
    both of them to a new column name with the objective to merge
    the dataframes with the preferred method based on this column.
    It also inserts the new column in the last position of the
    merged dataframe.
    If left is the chosen merge method, this function will keep
    all rows of df1 and delete all columns from df2 that had no
    merge.

    Parameters
    ----------
    df1 : Dataframe
        Dataframe that will be merge too.
    df2 : Dataframe
        Dataframe that will merge into.
    column_df1 : String
        Column from df1 used for the merge.
    column_df2 : String
        Column from df2 used for the merge.
    column_new : String
        Name of the new merged column that will be in the final dataframe.
    merge_type : String
        Type of merge to be performed.
        Choose between inner, outer, left, right, cross. Default is inner.

    Returns
    -------
    df_out : Dataframe
        Final merged dataframe.
    """
    
    # Renaming the columns that will merge to have the same name
    df1.rename(columns = {column_df1 : column_new}, inplace = True )
    df2.rename(columns = {column_df2 : column_new}, inplace = True )
    # Separating the dataframes in rows with nulls and not nulls so nulls dont merge
    df1_notn = df1[df1[column_new].notnull() & (df1[column_new] != 'nan')]
    df1_isn = df1[df1[column_new].isnull() | (df1[column_new] == 'nan')]
    df2_notn = df2[df2[column_new].notnull() & (df2[column_new] != 'nan')]
    df2_isn = df2[df2[column_new].isnull() | (df2[column_new] == 'nan')]
    # The actual merge
    df_out = df1_notn.merge(df2_notn, how = merge_type, on = column_new)
    # Concatanating the not nulls merged with nulls
    df_out = pd.concat([df_out, df1_isn, df2_isn])
    # Renaming the columns used for merge back to the original
    df1.rename(columns = {column_new : column_df1}, inplace = True )
    df2.rename(columns = {column_new : column_df2}, inplace = True )
    # Putting the new merged column in the end of the dataframe
    column_pop = df_out.pop(column_new)
    df_out.insert(len(df_out.columns), column_new, column_pop)
    # Taking all the blanks in right if a left merge was used
    if merge_type == 'left':        
        col = df_out.pop(column_new)
        df_out.insert(len(df1.columns)-1, column_new, col)
        df_out = df_out[df_out[df_out.columns[len(df1.columns) + 1]].isnull()]
        df_out = df_out.drop(df_out.columns[-len(df2.columns) + 1:], axis=1)
    return df_out


#%% =============== MERGE VALIDATION ===============  


def count_cell_occurence(df, column):
    """
    This function counts the number of occurences of each value in
    the specified column adding a new column named counts with the
    number of occurences.

    Parameters
    ----------
    df : Dataframe
        Dataframe that will be used for the count.
    column : TYPE
        Column that will be counted.

    Returns
    -------
    None.

    """
    counts = df[column].value_counts()
    df['count'] = df[column].map(counts)

    
def fuzzy_row_between_two_columns(df, column1, column2, new_column):
    df[new_column] = df.apply(lambda x: fuzz.token_sort_ratio(x[column1], x[column2]), axis=1)
    

def name_token(name,ind):
    # ind = 0 first name
    # ind = 1 second name
    # ind = 2 third name
    # ind = -1 last name
    try :
        return name.split()[ind]
    except:
        return ''


def validate_names(df, name1, name2):
    # disable chained assignments
    pd.options.mode.chained_assignment = None 
    
    df['First and Last']    = False
    df['First and Second']  = False
    df['Last and Second']   = False
    df['Second and Last']   = False
    df['First and Third']   = False
    df['Last and Third']    = False
    df['Third and Last']    = False
    df['Second and Third']  = False
    df['Third and Second']  = False
    pd.options.mode.chained_assignment = 'warn' 
    
    for i in df.index:
        name = df.at[i,name1]
        first_name1  = name_token(name,0)
        second_name1 = name_token(name,1)
        third_name1  = name_token(name,2)
        last_name1   = name_token(name,-1)

        name = df.at[i,name2]
        first_name2  = name_token(name,0)
        second_name2 = name_token(name,1)
        third_name2  = name_token(name,2)
        last_name2   = name_token(name,-1)

        #if name:        
        if first_name1 == first_name2:
            df.at[i,'First and Last']   = (last_name1   == last_name2)
            df.at[i,'First and Second'] = (second_name1 == second_name2)
            df.at[i,'Last and Second']  = (last_name1   == second_name2)
            df.at[i,'Second and Last']  = (second_name1 == last_name2)
            df.at[i,'First and Third']  = (third_name1  == third_name2)
            df.at[i,'Last and Third']   = (last_name1   == third_name2)
            df.at[i,'Third and Last']   = (third_name1  == last_name2)
            df.at[i,'Second and Third'] = (second_name1 == third_name2)
            df.at[i,'Third and Second'] = (third_name1  == second_name2)


def split_validated_and_non(df):
    df_out1 = df[(df['First and Last'] == True) | (df['First and Second'] == True) | (df['Last and Second'] == True) | (df['Second and Last'] == True) | (df['First and Third'] == True) | (df['Last and Third'] == True) | (df['Third and Last'] == True) | (df['Second and Third'] == True) | (df['Third and Second'] == True)]
    df_out2 = df[~((df['First and Last'] == True) | (df['First and Second'] == True) | (df['Last and Second'] == True) | (df['Second and Last'] == True) | (df['First and Third'] == True) | (df['Last and Third'] == True) | (df['Third and Last'] == True) | (df['Second and Third'] == True) | (df['Third and Second'] == True))]
    df_out1 = df_out1.drop(df_out1.columns[-9:], axis = 1)
    df_out2 = df_out2.drop(df_out2.columns[-9:], axis = 1)
    return df_out1, df_out2


def validate_score(df1, df2, column, score):
    df1 = pd.concat([df1, df2[df2[column] >= score]])
    df2 = df2[df2[column] < score]
    return df1, df2


def send_dup_ref_val_to_not_val(dfval, dfnval, column_index1, column_index2, column_fuzzy):
    # Get the duplicates in the index 1 column
    duplicates = dfval[dfval.duplicated(subset=column_index1, keep=False)]
    # Group the duplicates by index 1 and get the index of the rows with the highest and lowest 'fuzzy_name' value
    grouped = duplicates.groupby(column_index1)
    lowest_index = grouped[column_fuzzy].idxmin()
    lowest = dfval.loc[lowest_index]
    # Drop the lowest 'fuzzy_name' values from the original dataframe
    dfval.drop(lowest_index, inplace=True)
    dfnval = pd.concat([dfnval, lowest])

    # Get the duplicates in the index 2 column
    duplicates = dfval[dfval.duplicated(subset=column_index2, keep=False)]
    # Group the duplicates by index 2 and get the index of the rows with the highest and lowest 'fuzzy_name' value
    grouped = duplicates.groupby(column_index2)
    lowest_index = grouped[column_fuzzy].idxmin()
    lowest = dfval.loc[lowest_index]
    # Drop the lowest 'fuzzy_name' values from the original dataframe
    dfval.drop(lowest_index, inplace=True)
    dfnval = pd.concat([dfnval, lowest])
    if dfval[dfval.duplicated(subset=column_index1, keep=False)].shape[0] > 0:
        send_dup_ref_val_to_not_val(dfval, dfnval, column_index1, column_index2, column_fuzzy)
    return dfval, dfnval


def delete_indexes_contained_in_val(dfval, dfnval, column_index1, column_index2, df_org_index1, df_org_index2):
    # Changes every value that came from df1 that have the index1 contained in validated df to nan.np
    df_temp = dfnval[dfnval[column_index1].isin(dfval[column_index1])]
    indexes_to_nan = df_temp.index
    dfnval.loc[indexes_to_nan, df_org_index1.columns] = np.nan
    # Changes every value that came from df2 that have the index2 contained in validated df to nan.np
    df_temp = dfnval[dfnval[column_index2].isin(dfval[column_index2])]
    indexes_to_nan = df_temp.index
    dfnval.loc[indexes_to_nan, df_org_index2.columns] = np.nan
    # Deletes the rows that have index1 and index2 contained in the validated df
    df_temp = dfnval[(dfnval[column_index1].isna()) & (dfnval[column_index2].isna())]
    indexes_to_delete = df_temp.index
    dfnval.drop(indexes_to_delete, inplace=True)
    return dfval, dfnval

def final_cleaning(dfnval, column_index1, column_index2, df_org_index1, df_org_index2):
    # Changes every value that came from df2 to nan.np
    df_temp1 = dfnval.copy()
    df_temp1.loc[:, df_org_index2.columns] = np.nan
    # Deletes the rows that have index1 and index2 empty
    df_temp = df_temp1[(df_temp1[column_index1].isna()) & (df_temp1[column_index2].isna())]
    indexes_to_delete = df_temp.index
    df_temp1.drop(indexes_to_delete, inplace=True)
    # Deletes the remaining duplicates that came from df1
    df_temp1.drop_duplicates(subset=[column_index1], keep='first', inplace=True)
    # Changes every value that came from df1 to nan.np
    df_temp2 = dfnval.copy()
    df_temp2.loc[:, df_org_index1.columns] = np.nan
    # Deletes the rows that have index1 and index1 empty
    df_temp = df_temp2[(df_temp2[column_index1].isna()) & (df_temp2[column_index2].isna())]
    indexes_to_delete = df_temp.index
    df_temp2.drop(indexes_to_delete, inplace=True)
    # Deletes the remaining duplicates that came from df2
    df_temp2.drop_duplicates(subset=[column_index2], keep='first', inplace=True)
    # Concates both sides of the dataframe
    dfnval = pd.concat([df_temp1, df_temp2])
    return dfnval


def final_configuration(dfval, dfnval, column_index1, column_index2, df_org_index1, df_org_index2, column_fuzzy, column_merged, name_org1, name_org2):
    dftotal = pd.concat([dfval, dfnval])
    df1_filtered = df_org_index1[~df_org_index1[column_index1].isin(dftotal[column_index1])]
    df2_filtered = df_org_index2[~df_org_index2[column_index2].isin(dftotal[column_index2])]
    dftotal = pd.concat([dftotal, df1_filtered, df2_filtered])
    dftotal = dftotal.iloc[:, :-2]
    dftotal['Name_merged_sort'] = dftotal[name_org1].where(dftotal[name_org1].notnull(), dftotal[name_org2])
    delete_double_spaces(dftotal, 'Name_merged_sort')
    lower_letters(dftotal, 'Name_merged_sort')
    dftotal = dftotal.sort_values(by=['Name_merged_sort'])
    #dftotal = dftotal.sort_values(by=[column_to_sort1, column_to_sort2])
    df_temp = dftotal[(dftotal[column_index1].isna()) | (dftotal[column_index2].isna())]
    indexes_to_nan = df_temp.index
    dftotal.loc[indexes_to_nan, [column_fuzzy, 'count', column_merged]] = np.nan
    # Resets index
    dftotal.reset_index()
    return dftotal
    
    
#%% =============== FUNCTIONS FOR CHECKING DATA ===============  
        
        
def list_of_chars(df, column):
    """
    This function creates a set with all unique chars used in the
    specified column.

    Parameters
    ----------
    df : Dataframe
        Dataframe that will be used for the set.
    column : String
        Column that will be used for the set.

    Returns
    -------
    Set
        Unique chars set.

    """
    return sorted(list(set(df[column].apply(list).sum())))


def filter_df_by_criterion(df, column):
    """
    This function creates a dataframe filtered by cells
    containing numbers, but the code can be used to filter it by
    any criteria needed. The code can be found below and only 
    needs to change the pattern in parentheses:
        df[df[column].str.contains(r'\d')]

    Parameters
    ----------
    df : Dataframe
        Dataframe that will be used for the filter.
    column : String
        Column that will be used for the filter.

    Returns
    -------
    Dataframe
        Filtered dataframe.

    """
    return df[df[column].str.contains(r'\d')]


def filter_list_by_criterion(df, column, column_index):
    """
    This function creates a list filtered by cells containing
    numbers, but the code can be used to filter it by any
    criteria needed. The code can be found below and only 
    needs to change the pattern in parentheses:
        [str(x[column_index]) for x in df[df[column].str.contains(r'\d')].values.tolist()]

    Parameters
    ----------
    df : Dataframe
        Dataframe that will be used for the filter.
    column : String
        Column that will be used for the filter.
    column_index : TYPE
        Index of the column that will be used for the filter.

    Returns
    -------
    List
        Filtered list.

    """
    return [str(x[column_index]) for x in df[df[column].str.contains(r'\d')].values.tolist()]


def count_chars_in_column(df, column):
    """
    This function prints the count of all lengths of the
    specified column.

    Parameters
    ----------
    df : Dataframe
        Dataframe that will be used for the count.
    column : String
        Column that will be used for the count.

    Returns
    -------
    None.

    """
    dfcount = df[column].str.len()
    print(dfcount.value_counts())


#%% =============== COMPILATIONS OF FUNCTIONS ===============


def cpf_cleaning(df, column_cpf):
    """
    This function was created to compile all functions used for
    CPF cleaning.

    Parameters
    ----------
    df : Dataframe
        Dataframe that will be cleaned.
    column_cpf : String
        CPF column that will be cleaned.

    Returns
    -------
    None.

    """
    # Changing CPF columns type to string
    column_to_string(df, column_cpf)
    # Deletes all non numeric chars from CPF
    delete_non_numeric(df, column_cpf)
    # Adding zeros for litify CPFs with length lower then 11
    add_zeros_to_cpf(df, column_cpf)
    # Replacing 00000000000 for NaN
    replace_zeroed_cpfs(df, column_cpf)


def name_cleaning(df, column_name):
    """
    This function was created to compile all functions used for
    name cleaning.

    Parameters
    ----------
    df : Dataframe
        Dataframe that will be cleaned.
    column_name : String
        Name column that will be cleaned.

    Returns
    -------
    None.

    """
    # Changing CPF columns type to string
    column_to_string(df, column_name)
    # Replacing accents of name
    replace_accents(df, column_name)
    # Lowering all letters of name
    lower_letters(df, column_name)
    # Taking out zeros after space of name
    zeros_after_spaces(df, column_name)
    # Replacing zeros for O of name
    zero_for_o(df, column_name)
    # Removing duplicate letters of name
    apply_func_letters(df, column_name)
    # Removing non alphabetic chars of name
    delete_non_alphabetic(df, column_name)
    # Replace multiple spaces with a single space of name
    delete_double_spaces(df, column_name)
    
    
def address_cleaning(df, column_address):
    """
    This function was created to compile all functions used for
    address cleaning.

    Parameters
    ----------
    df : Dataframe
        Dataframe that will be cleaned.
    column_address : String
        Address column that will be cleaned.

    Returns
    -------
    None.

    """
    # Changing CPF columns type to string
    column_to_string(df, column_address)
    # Replacing accents of address
    replace_accents(df, column_address)
    # Lowering all letters of address
    lower_letters(df, column_address)
    # Removing letter duplicates of address
    apply_func_letters(df, column_address)
    # Changing street details of address
    apply_func_street(df, column_address)
    # Changing address details of address
    apply_func_address(df, column_address)
    # Removing non alphabetic and numeric chars of address
    delete_non_alphanumeric(df, column_address)
    # Deleting postcode of address
    delete_postcode(df, column_address)
    # Replace multiple spaces with a single space of address
    delete_double_spaces(df, column_address)


def cleaning_with_cpf(df, column_cpf, column_name, column_address):
    """
    This function was created to compile all functions used for
    cleaning in general when the dataframe have CPF.

    Parameters
    ----------
    df : Dataframe
        Dataframe that will be cleaned.
    column_cpf : String
        CPF column that will be cleaned.
    column_name : String
        Name column that will be cleaned.
    column_address : String
        Address column that will be cleaned.

    Returns
    -------
    None.

    """
    cpf_cleaning(df, column_cpf)
    name_cleaning(df, column_name)
    address_cleaning(df, column_address)
    

def cleaning_without_cpf(df, column_name, column_address):
    """
    This function was created to compile all functions used for
    cleaning in general when the dataframe dont have CPF.

    Parameters
    ----------
    df : Dataframe
        Dataframe that will be cleaned.
    column_name : String
        Name column that will be cleaned.
    column_address : String
        Address column that will be cleaned.

    Returns
    -------
    None.

    """
    name_cleaning(df, column_name)
    address_cleaning(df, column_address)
    

def merging_by_cpf_validating_by_name(df1, df2, column_cpf1, column_cpf2, column_name1, column_name2, column_address1, column_address2, column_merged_cpf, column_merged_name, merge_type, column_index1, column_index2):
    # Merge by CPF
    dfmerge = merge_dataframes(df1, df2, column_cpf1, column_cpf2, column_merged_cpf, merge_type)
    column_to_string(dfmerge, column_name1)
    column_to_string(dfmerge, column_name2)
    # Taking out CPFs that were merged but both names are nan, getting 100 of fuzzy
    dfmerge_notnulls = dfmerge[~((dfmerge[column_name1] == 'nan') | (dfmerge[column_name2] == 'nan') | (dfmerge[column_name1] == '') | (dfmerge[column_name1] == ''))]
    dfmerge_nulls = dfmerge[(dfmerge[column_name1] == 'nan') | (dfmerge[column_name2] == 'nan') | (dfmerge[column_name1] == '') | (dfmerge[column_name1] == '')]
    dfmerge = dfmerge_notnulls
    # Create a column for the number of occurences of each cell in its column
    count_cell_occurence(dfmerge, column_merged_cpf)
    # Fuzzy score by row between the columns of name
    fuzzy_row_between_two_columns(dfmerge, column_name1, column_name2, 'fuzzy_name')
    # Separate dataframe between name score higher or equal then 80 and lower
    dfmerge_80_higher = dfmerge[(dfmerge['fuzzy_name'] >= 80) & (dfmerge[column_name1].apply(lambda x: str(x).split()[0] if str(x).split() else "") == dfmerge[column_name2].apply(lambda x: str(x).split()[0] if str(x).split() else ""))]
    dfmerge_80_lower = dfmerge[~((dfmerge['fuzzy_name'] >= 80) & (dfmerge[column_name1].apply(lambda x: str(x).split()[0] if str(x).split() else "") == dfmerge[column_name2].apply(lambda x: str(x).split()[0] if str(x).split() else "")))]
    # Creating columns for validation of lower scores
    validate_names(dfmerge_80_lower, column_name1, column_name2)
    # Spliting the dataframe between validated and not validated
    dfmerge_validated, dfmerge_not_validated = split_validated_and_non(dfmerge_80_lower)
    # Adding higher scores to final validated and nulls final not validated
    dfmerge_validated = pd.concat([dfmerge_validated, dfmerge_80_higher])
    dfmerge_not_validated = pd.concat([dfmerge_not_validated, dfmerge_nulls])
    
    # Validating the not validated with merge on name and address score higher or equal then 80
    dfnval1 = dfmerge_not_validated[(dfmerge_not_validated[column_index1].isna())].dropna(how='all', axis=1).drop(columns=[column_name1])
    dfnval2 = dfmerge_not_validated[(dfmerge_not_validated[column_index2].isna())].dropna(how='all', axis=1).drop(columns=[column_name2])
    dfnval3 = dfmerge_not_validated[~((dfmerge_not_validated[column_index1].isna()) | (dfmerge_not_validated[column_index2].isna()))]
    # Merge by name
    dfmerge = merge_dataframes(dfnval2, dfnval1, column_name1, column_name2, column_merged_name, merge_type)
    column_to_string(dfmerge, column_address1)
    column_to_string(dfmerge, column_address2)
    # Taking out CPFs that were merged but both names are nan, getting 100 of fuzzy
    dfmerge_notnulls = dfmerge[~((dfmerge[column_address1] == 'nan') | (dfmerge[column_address2] == 'nan') | (dfmerge[column_address1] == '') | (dfmerge[column_address2] == ''))]
    dfmerge_nulls = dfmerge_nulls = dfmerge[(dfmerge[column_address1] == 'nan') | (dfmerge[column_address2] == 'nan') | (dfmerge[column_address1] == '') | (dfmerge[column_address2] == '')]
    dfmerge = dfmerge_notnulls
    # Create a column for the number of occurences of each cell in its column
    count_cell_occurence(dfmerge, column_merged_name)
    # Fuzzy score by row between the columns of address
    fuzzy_row_between_two_columns(dfmerge, column_address1, column_address2, 'fuzzy_address')
    # Separate dataframe between name score higher or equal then 80 and lower
    dfmerge_80_higher = dfmerge[dfmerge['fuzzy_address'] >= 80]
    dfmerge_80_lower = dfmerge[dfmerge['fuzzy_address'] < 80]
    # Adding higher scores to final validated and nulls final not validated
    dfmerge_validated = pd.concat([dfmerge_validated, dfmerge_80_higher])
    dfmerge_not_validated = pd.concat([dfmerge_80_lower, dfmerge_nulls, dfnval3])
    # Replace nan values in fuzzy_name column with 0 for comparison on send_dup_ref_val_to_not_val
    dfmerge_validated['fuzzy_name'] = dfmerge_validated['fuzzy_name'].replace(np.nan, 0)

    
    dfmerge_validated, dfmerge_not_validated = send_dup_ref_val_to_not_val(dfmerge_validated, dfmerge_not_validated, column_index1, column_index2, 'fuzzy_name')
    dfmerge_validated, dfmerge_not_validated = delete_indexes_contained_in_val(dfmerge_validated, dfmerge_not_validated, column_index1, column_index2, df1, df2)
    dfmerge_not_validated =  final_cleaning(dfmerge_not_validated, column_index1, column_index2, df1, df2)
    
    return dfmerge_validated, dfmerge_not_validated       
    
    
def merging_by_name_validating_by_address(df1, df2, column_name1, column_name2, column_address1, column_address2, column_merged, merge_type, column_index1, column_index2):
    # Merge by name
    dfmerge = merge_dataframes(df1, df2, column_name1, column_name2, column_merged, merge_type)
    column_to_string(dfmerge, column_address1)
    column_to_string(dfmerge, column_address2)
    # Taking out CPFs that were merged but both names are nan, getting 100 of fuzzy
    dfmerge_notnulls = dfmerge[~((dfmerge[column_address1] == 'nan') | (dfmerge[column_address2] == 'nan') | (dfmerge[column_address1] == '') | (dfmerge[column_address2] == ''))]
    dfmerge_nulls = dfmerge_nulls = dfmerge[(dfmerge[column_address1] == 'nan') | (dfmerge[column_address2] == 'nan') | (dfmerge[column_address1] == '') | (dfmerge[column_address2] == '')]
    dfmerge = dfmerge_notnulls
    # Create a column for the number of occurences of each cell in its column
    count_cell_occurence(dfmerge, column_merged)
    # Fuzzy score by row between the columns of address
    fuzzy_row_between_two_columns(dfmerge, column_address1, column_address2, 'fuzzy_address')
    # Separate dataframe between name score higher or equal then 80 and lower
    dfmerge_validated = dfmerge[dfmerge['fuzzy_address'] >= 80]
    dfmerge_not_validated = dfmerge[dfmerge['fuzzy_address'] < 80]
    # Adding higher scores to final validated and nulls final not validated
    dfmerge_not_validated = pd.concat([dfmerge_not_validated, dfmerge_nulls])
    
    #send_dup_ref_val_to_not_val(dfmerge_validated, dfmerge_not_validated, column_index1, column_index2, 'fuzzy_address')
    
    return dfmerge_validated, dfmerge_not_validated

def to_pickle(df,file): 
    """
    This can 16x faster than pd.to_excel
    
    Parameters
    ----------
    df : pandas dataframe pickled 
        
    file : file to store the df pickled
    
    Returns
    -------
    None.

    """
    with open(file, 'wb') as f:
        pickle.dump(df, f)

def read_pickle(file): 
    """
    This can 28x faster than pd.to_excel
    
    Parameters
    ----------
        
    file : file whwre the dataframe is pickled
    
    Returns
    -------
    df : dataframe
    """
    with open(file, 'rb') as f:
        return pickle.load(f)