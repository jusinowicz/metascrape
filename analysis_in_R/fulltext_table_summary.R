#=============================================================================
# Using the table of words scraped from the results and methods of the 
# fulltext and combined with abstract scraping to generate tables.
#=============================================================================
# Load necessary libraries
library(dplyr)
#=============================================================================

# Read the ID key from 'all_DOIS.csv'
allDOIs = read.csv("all_DOIS.csv")
colnames(allDOIs)[2] = "STUDY"

# Read the abstrac data from 'abstract_parsing1.csv'
abstracts = read.csv('./../output/abstract_parsing1.csv')

# Read the fulltext data table from the output folder:
data = read.csv("./../output/extract_from_text1.csv")
#data$STUDY = as.numeric(data$STUDY)

# #Read the data table scraped from methods in the output folder:
methods_data = read.csv("./../output/extract_from_methods1.csv")

#=============================================================================
#Some cleaning
#=============================================================================

# Clean the CARDINAL column
data$CARDINAL = data$CARDINAL %>%
  # Replace any non-numeric characters (including letters and symbols) with NA
  gsub("[^0-9\\.]", "", .) %>%
  # Convert empty strings to NA
  na_if("") %>%
  # Convert to numeric
  as.numeric()
  
#Remove things that are dates or other odd numbers, anything over 1900: 
data$CARDINAL[data$CARDINAL>=900] = NA

#=============================================================================
#Data summaries
#=============================================================================
# Summarize CARDINAL: mean and median, excluding blanks (assuming blanks are represented by NA)
cardinal_summary = data %>%
  filter(!is.na(CARDINAL)) %>%
  summarise(
    mean_CARDINAL = mean(CARDINAL, na.rm = TRUE),
    median_CARDINAL = median(CARDINAL, na.rm = TRUE)
  )

# Count different RESPONSE types
response_counts = data %>%
  count(RESPONSE)

# Count unique STUDYs for which ISTABLE == 0
study_count = data %>%
  filter(ISTABLE == 0) %>%
  summarise(unique_studies = n_distinct(STUDY))

# Print the summaries
print(cardinal_summary)
print(response_counts)
print(study_count)

#=============================================================================
#Merge the data with some additional variables from abstracts and methods
#=============================================================================

# Merge high-quality data with 'allDOIs.csv' to get the 'DOI' column
data = data %>%
  inner_join(allDOIs, by = "STUDY")

#Add only the LAT/LON data from the methods
data = data %>%
  left_join(select(methods_data, DOI, LAT, LON), by = "DOI")

# Take the columns from the Abstracts that are not yet associated with the tables high-quality data with 'abstract_parsing1.csv' based on the 'DOI' column
merged_data = data %>%
  left_join(select(abstracts, DOI, INOCTYPE, SOILTYPE, FIELDGREENHOUSE, LANDUSE, ECOTYPE, ECOREGION, LOCATION), by = "DOI")

# Define criteria for high-quality data (example: ISTABLE == 0 and CARDINAL is not NA or text)
high_quality_data = merged_data %>%
  filter(ISTABLE == 0 & !is.na(CARDINAL))

# Display the high_quality data
print(head(high_quality_data))

#=============================================================================
# Save the combined sheet
write.csv(file = "./../output/fulltext_abstracts_comb.csv", high_quality_data) 


