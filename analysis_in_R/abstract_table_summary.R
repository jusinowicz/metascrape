#=============================================================================
# Using the table of words scraped from abstracts, abstract_parsing1.csv, 
# generate some word summaries for each label.
#
#=============================================================================
library(readr)
library(dplyr)
library(tidyr)
library(purrr)
library(ggplot2)
library(wordcloud2)
library(htmlwidgets)
library(pandoc)
#=============================================================================
# Load Data
#=============================================================================
ap_dir = "./../output/abstract_parsing1.csv"

# Read the CSV file
data = read_csv(ap_dir)

#=============================================================================
#Custom functions
#=============================================================================
# Function to split and unnest words
split_and_unnest = function(df, column_name) {
  df %>%
    mutate(across(all_of(column_name), \(x) strsplit(x, split = ";"))) %>%
    unnest({{ column_name }})
}

# Function to split phrases into a list of data frames
split_cols_list = function(df, column_name, study_id_col) {
  df %>%
    mutate({{ column_name }} := strsplit(.data[[column_name]], split = ";")) %>%
    unnest({{ column_name }}) %>%
    distinct() %>%
    select(study_id = all_of(study_id_col), phrase = all_of(column_name))
}


#Function to do this all in one step
list_and_unnest = function(df, column_name, study_id_col) {
    df %>%
      mutate(phrases = strsplit(.data[[column_name]], split = ";")) %>%
      unnest(phrases) %>%
      distinct() %>%
      select(study_id = all_of(study_id_col), phrase = phrases)
  }

# Function to summarize unique phrases and their counts
summarize_phrases = function(df) {
  df %>%
    count(phrase, name = "count") %>%  # Count occurrences
    arrange(desc(count))
}

#=============================================================================
#Main workflow
#=============================================================================
#Split each label column into a list, then unpack all of the phrases
#into new entries. 
study_id = names(data)[1]
column_names = names(data)[names(data) != study_id]
processed_data = map(column_names, function(col){
  list_and_unnest(data, col,study_id)
})

#Put the correct names back on the columns
for(n in 1:length(column_names)){ 
  colnames(processed_data[[n]])[2] = column_names[n]
}

# Summarize phrases for each data frame in the list
summaries = map(processed_data, summarize_phrases)

#=============================================================================
#Visualizing with ggplot
#=============================================================================
#This will use walk to cycle through the list, summaries
# Function to create a bar plot for a summary
create_bar_plot <- function(summary_df, column_name) {
  ggplot(summary_df, aes_string(x = "phrase", y = "count")) +
    geom_bar(stat = "identity") +
    coord_flip() + # Flip coordinates for better readability
    theme_minimal() +
    labs(title = paste("Frequency of Phrases in", column_name),
         x = "Phrase",
         y = "Count")
}
# Create and save bar plots for each summary
walk2(names(data_use), summaries, function(col, summary_df) {
  plot = create_bar_plot(summary_df, col)
  ggsave(paste0("./../output/bar_plot_", col, ".png"), plot, width = 8, height = 6)
})


#=============================================================================
#Word cloud
#=============================================================================
# Function to create a word cloud for a summary
create_word_cloud = function(summary_df, column_name) {
  wordcloud2(summary_df, size = 1, color = 'random-light', backgroundColor = "black")
}

# Create and save word clouds for each summary
walk2(names(data_use), summaries, function(col, summary_df) {
  # Wordcloud2 requires a proper filename for HTML
  html_filename = paste0("./../output/word_cloud_", col, ".html")
  
  # Save word cloud as HTML file without embedding all dependencies
  saveWidget(create_word_cloud(summary_df, col), html_filename, selfcontained = FALSE)
})
