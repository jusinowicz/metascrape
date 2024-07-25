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

# Function to summarize unique phrases and their counts
summarize_phrases = function(df, column_name) {
  df %>%
    group_by(.data[[column_name]]) %>%
    summarise(count = n()) %>%
    ungroup() %>%
    arrange(desc(count))
}


#=============================================================================
#Main workflow
#=============================================================================
# Assume the first column is "study ID" and should not be processed
data_use = data[,-1] 
# Process each column dynamically
processed_data = reduce(names(data_use), function(df, col) {
  split_and_unnest(df, col)
}, .init = data_use)

# Add the study ID column back to the processed data
processed_data = bind_cols(data[,1], processed_data)

# Summarize phrases for each column except the study ID column
summaries = map(names(data_use), function(col) {
  summarize_phrases(processed_data, col)
})


#=============================================================================
#Visualizing with ggplot
#=============================================================================
#This will use walk to cycle through the 
# Function to create a bar plot for a summary
create_bar_plot <- function(summary_df, column_name) {
  ggplot(summary_df, aes_string(x = column_name, y = "count")) +
    geom_bar(stat = "identity") +
    coord_flip() + # Flip coordinates for better readability
    theme_minimal() +
    labs(title = paste("Frequency of Phrases in", column_name),
         x = column_name,
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
