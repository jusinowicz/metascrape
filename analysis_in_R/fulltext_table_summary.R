# Load necessary libraries
library(dplyr)

# Read the ID key from 'all_DOIS.csv'
allDOIs = read.csv("all_DOIS.csv")
colnames(allDOIs)[2] = "STUDY"

# Read the abstrac data from 'abstract_parsing1.csv'
abstracts = read.csv('./../output/abstract_parsing1.csv')

# Read the data table from the output folder:
data = read.csv("./../output/extract_from_text2.csv")
data$STUDY = as.numeric(data$STUDY)

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

# Define criteria for high-quality data (example: ISTABLE == 0 and CARDINAL is not NA or text)
high_quality_data = data %>%
  filter(ISTABLE == 0 & !is.na(CARDINAL))


# Merge high-quality data with 'allDOIs.csv' to get the 'DOI' column
merged_data = high_quality_data %>%
  inner_join(allDOIs, by = "STUDY")

# Merge high-quality data with 'abstract_parsing1.csv' based on the 'DOI' column
merged_data = merged_data %>%
  inner_join(abstracts, by = "DOI")

# Display the merged data
print(head(merged_data))
 
# Save the combined sheet
write.csv(file = "./../output/fulltext_abstracts_comb.csv", merged_data) 