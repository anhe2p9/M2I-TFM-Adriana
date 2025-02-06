# Import the required libraries
from typing import List
import pandas as pd
from ILP_data_from_refactoring_cache.utils import offsets as offsets


# Filter feasible extractions and set extractions id (from 0 to N)
#
# This function assumes the input dataframe has the following columns: A, B, feasibility, reason, parameters,
# extractedLOC, reductionCC, extractedMethodCC, accumulatedInherentComponent, accumulatedNestingComponent,
# numberNestingContributors, nesting
def set_extractions_id(df: pd.DataFrame) -> pd.DataFrame:
    # Filter the DataFrame by feasible extractions
    filtered_df = df[df['feasible'] == 1]

    # Sort the DataFrame by the start offset (ascending) and the end offset (descending)
    df_sorted = filtered_df.sort_values(by=[filtered_df.columns[0], filtered_df.columns[1]], ascending=[True, False])

    # Add a new column with extractions id (from 0 to N)
    df_sorted.insert(0, 'extraction', range(len(df_sorted)))

    return df_sorted


# Return a dataframe with the nested extractions, including the ancestor extraction and additional columns from the
# original dataset
#
# This function assumes the input dataframe has the following columns: extraction, A, B, feasibility, reason,
# parameters, extractedLOC, reductionCC, extractedMethodCC, accumulatedInherentComponent,
# accumulatedNestingComponent, numberNestingContributors, nesting
#
# Note that the first column can be computed by the set_extractions_id function
def get_nested_extraction_for_each_extraction(df: pd.DataFrame, columns_to_add: List[str]) -> pd.DataFrame:
    # Create a set of tuples (nested extraction, ancestor extraction, ...) to store the nested extractions
    nested_extractions = set()

    # Loop to process each row of the DataFrame
    for index1 in range(0, len(df)):
        row1 = df.iloc[index1]
        # Process the current row (row1)
        offset1 = (row1['A'], row1['B'])

        # Loop to process the rows after the current row
        for index2 in range(index1 + 1, len(df)):
            row2 = df.iloc[index2]
            offset2 = (row2['A'], row2['B'])

            # Example condition to identify a nested extraction
            if offsets.is_contained(offset2, offset1):
                extraction1 = row1['extraction']
                extraction2 = row2['extraction']
                tuple_to_add = (extraction2, extraction1) + tuple(row2[c] for c in columns_to_add)
                nested_extractions.add(tuple_to_add)

    # Move the nested extractions to a DataFrame
    result = pd.DataFrame(list(nested_extractions), columns=['NestedExtraction', 'AncestorExtraction'] + columns_to_add)

    # Sort the DataFrame by the second and first columns
    result = result.sort_values(by=['AncestorExtraction', 'NestedExtraction'], ascending=[True, True])

    return result


# Return a dataframe with the nested extractions, including the ancestor extraction and the cognitive complexity
# reduction (CCR) removed from the ancestor when removing a nested extraction
#
# This function assumes the input dataframe has the following columns: extraction, A, B, feasibility, reason,
# parameters, extractedLOC, reductionCC, extractedMethodCC, accumulatedInherentComponent,
# accumulatedNestingComponent, numberNestingContributors, nesting
#
# Note that the first column can be computed by the set_extractions_id function
def get_nested_extraction_for_each_extraction_computing_ccr(df: pd.DataFrame) -> pd.DataFrame:
    # Create a set of tuples (nested extraction, ancestor extraction, CCR)
    nested_extractions = set()

    # Loop to process each row of the DataFrame
    for index1 in range(0, len(df)):
        row1 = df.iloc[index1]
        # Process the current row (row1)
        offset1 = (row1['A'], row1['B'])

        # Loop to process the rows after the current row
        for index2 in range(index1 + 1, len(df)):
            row2 = df.iloc[index2]
            offset2 = (row2['A'], row2['B'])

            # Example condition to identify a nested extraction
            if offsets.is_contained(offset2, offset1):
                i_extraction = row1['extraction']  # the ancestor extraction
                j_extraction = row2['extraction']  # the nested extraction
                CCRji = row2['accumulatedInherentComponent'] + row2['accumulatedNestingComponent'] + abs(
                    row2['nesting'] - row1['nesting']) * row2['numberNestingContributors']
                tuple_to_add = (j_extraction, i_extraction, CCRji)
                nested_extractions.add(tuple_to_add)

    # Move the nested extractions to a DataFrame
    result = pd.DataFrame(list(nested_extractions), columns=['NestedExtraction', 'AncestorExtraction', 'CCR'])

    # Sort the DataFrame by the second and first columns
    result = result.sort_values(by=['AncestorExtraction', 'NestedExtraction'], ascending=[True, True])

    return result


# Return a dataframe with the extractions in conflict
#
# This function assumes the input dataframe has the following columns: extraction, A, B, feasibility, reason,
# parameters, extractedLOC, reductionCC, extractedMethodCC, accumulatedInherentComponent,
# accumulatedNestingComponent, numberNestingContributors, nesting
#
# Note that the first column can be computed by the set_extractions_id function
def get_conflicts(df: pd.DataFrame) -> pd.DataFrame:
    conflicts = set()

    # Loop to process each row of the DataFrame
    for index1 in range(0, len(df)):
        row1 = df.iloc[index1]
        # Process the current row (row1)
        offset1 = (row1['A'], row1['B'])

        # Loop to process the rows after the current row
        for index2 in range(index1 + 1, len(df)):
            row2 = df.iloc[index2]
            offset2 = (row2['A'], row2['B'])

            # Example condition to identify a conflict
            if offsets.is_in_conflict(offset1, offset2):
                extraction1 = row1['extraction']
                extraction2 = row2['extraction']
                conflicts.add((extraction1, extraction2))

    # Move the conflicts to a DataFrame
    result = pd.DataFrame(list(conflicts), columns=['ExtractionA', 'ExtractionB'])

    # Sort the DataFrame by the first and second columns
    result = result.sort_values(by=['ExtractionA', 'ExtractionB'], ascending=[True, True])

    return result
