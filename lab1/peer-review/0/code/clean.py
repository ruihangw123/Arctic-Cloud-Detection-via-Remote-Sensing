##################
## clean_data() ##
##################

import numpy as np

def clean_data(df, 
               GCS_cutoff = 14, 
               split = False,
               replace = {90: 0,
                          91: np.nan,
                          92: np.nan}
                ):
    """
    Takes in the raw pandas dataframe from the TBI-PUD csv
    and cleans the data according to the following process:
    1. Remove Data with GSC scores lower than the cutoff set
        by the user. Use GSCGroup to filter if score GSC_cutoff
        is 14, use GCSTotal otherwise. Remove data that is 
        missing primary outcome.
    2. Remove GCSTotal Feature, can always recover it later
        for a plot, but do not need a linear combination of
        features when doing modelling.
    3. Remove AgeinYears feature, AgeInMonths is the more
        fine measurement
    4. Add pre-verbal/non-verbal as its own feature:
        0 is "pre-verbal/non-verbal," 1 is "responsive".
    5. Any feature with 90, 91, or 92, replace with user
        specified value. 

    Inputs:
        df: pandas dataframe to be cleaned
        GSC_cutoff: double/integer The largest GSC score to be included
            in the cleaned data. default is 14.
        split: Boolean, describes if the function should split the 
            clean dataframe in two. One for children aged 2 and above,
            one for children younger than 2.
        replace: dictionary, contains values to replace special codes
            with. By default, replaces them all with np.nan.

    Outputs:
        cleandf: cleaned pandas dataframe. If split is True, then contains only 
            the data for patients at least two years old.
        cleandf_young: optional, the second cleaned dataframe that is a result of
            the split parameter. Contains the data for patients under the age
            of 2 years.
    """

    # careful not to change the original dataframe the dataframes
    # we are working with should be small enough to store
    # two copies locally.
    cleandf = df.copy()

    # 1. remove rows with low GCS and missing primary outcome (PosIntFinal)
    cleandf = cleandf[~((cleandf["GCSTotal"] < GCS_cutoff) & (cleandf["PosIntFinal"].isnull()))]

    # 2. Create verbal indicator variable
    cleandf["verbal"] = (
        (cleandf["HA_verb"].fillna(-1) != 91) & 
        (cleandf["Amnesia_verb"].fillna(-1) != 91)
        ).astype(float)

    # 3. replace special codes with 0, NaN, and NaN
    cleandf = cleandf.replace(replace)

    # 4. check there are not inconsistencies with dependent columns
    # enforce() function is defined below
    cleandf= enforce(cleandf)

    # 5. normalize column names
    cleandf.columns = [c.strip().lower().replace("_", "") for c in cleandf.columns]

    # split the data if prompted
    if split:
        cleandf_young = cleandf[cleandf["agetwoplus"] == 1].drop("agetwoplus", inplace = True)
        cleandf = cleandf[cleandf["agetwoplus"] == 2].drop("agetwoplus", inplace = True)
        return cleandf, cleandf_young
    else:
        return cleandf
    
#####################
## helper function ##
#####################
    
def enforce(df, dependencies = 
            [("LOCSeparate", "LOCSeparate", "LocLen"),
            ("Seiz", "Seiz", "SeizLen"),
            ("HA_verb", "HA_verb", "HAStart"),
            ("Vomit", "Vomit", "VomitLast"),
            ("AMS", "AMS", "AMSOth"),
            ("SFxBas", "SFxBas", "SFxBasRhi"),
            ("SFxPalp", "SFxPalp", "SFxPalpDepress"),
            ("Hema", "Hema", "HemaSize"),
            ("Clav", "Clav", "ClavTem"),
            ("NeuroD", "NeuroD", "NeuroDOth"),
            ("OSI", "OSI", "OSIOth"),
            ("CTForm1", "CTForm1", "CTSedOth"),
            ("CTDone", "CTDone", "Finding23"),]
            ):
    """
    Takes in a data frame and a list of dependencies and forces the
    dependent columns to be zero whenever the parent column is 0.

    inputs:
        df: pandas dataframe
        dependencies: a list of tuples containing the parent feature, and the
            start and end of the dependent featurs in the dataframe.

    outputs:
        df: pandas dataframe with dependencies enforced.    
    
    """
    
    for parent_col, start_col, end_col in dependencies:

        # Find rows where parent is 0
        mask = df[parent_col] == 0

        # Set all dependent columns to 0 where parent is 0
        df.loc[mask, start_col:end_col] = 0

    return df