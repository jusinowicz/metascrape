#=============================================================================
# Using the table of words scraped from abstracts, abstract_parsing1.csv, 
# generate some word summaries for each label.
#
#=============================================================================
library(readr)
library(dplyr)
library(tidyr)
library(purrr)
library(stringr)

#For plots
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

#=============================================================================
#Function to do this all in one step
list_and_unnest = function(df, column_name, study_id_col) {
    df %>%
      mutate(phrases = strsplit(tolower(.data[[column_name]]), split = ";")) %>%
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
# Function to combine phrases that share common words directly
combine_phrases = function(df) {
  # Tokenize phrases into words
  df_words = df %>%
    mutate(words = strsplit(phrase, split = " ")) %>%
    unnest(words) %>%
    filter(words != "")
  
  # Create a mapping of each word to its corresponding phrases
  word_phrase_mapping = df_words %>%
    select(words, phrase) %>%
    distinct() %>%
    group_by(words) %>%
    summarise(phrases = list(unique(phrase))) %>%
    ungroup()

  # Initialize a list to store combined phrases
  combined_phrases_list = list()
  seen_phrases = character()
  
  # Combine phrases that share common words
  for (i in seq_along(word_phrase_mapping$phrases)) {
    current_phrases = word_phrase_mapping$phrases[[i]]
    if (!any(current_phrases %in% seen_phrases)) {
      combined_phrases_list = c(combined_phrases_list, list(unique(current_phrases)))
      seen_phrases = c(seen_phrases, current_phrases)
    }
  }

  # Create a data frame for combined phrases
  combined_phrases_df = tibble(
    combined_phrases = sapply(combined_phrases_list, paste, collapse = "; "),
    phrases = combined_phrases_list
  )

  # Summarize counts of combined phrases
  combined_df = df %>%
    rowwise() %>%
    mutate(combined_phrase = combined_phrases_df$combined_phrases[
      sapply(combined_phrases_df$phrases, function(p) phrase %in% p) %>% which.max()
    ]) %>%
    ungroup() %>%
    group_by(combined_phrase) %>%
    summarise(count = sum(count)) %>%
    arrange(desc(count)) %>%
    select(phrase = combined_phrase, count)
  
  combined_df
}

#=============================================================================
#Main workflow
#=============================================================================
#Split each label column into a list, then unpack all of the phrases
#into new entries. 
study_id = names(data)[1]
column_names = names(data)[names(data) != study_id]

#This will unpack each word/phrase and create a long list where each
#word/phrase is matched with the Study ID.
processed_data = map(column_names, function(col){
  list_and_unnest(data, col,study_id)
})

#Put the correct names back on the columns
for(n in 1:length(column_names)){ 
  colnames(processed_data[[n]])[2] = column_names[n]
}

# Create summary counts for each data frame in the list.
# This attemps to group similar words/phrases into a single 
# count item. 
summaries = vector("list", length(column_names))
for (n in 1:length(column_names)){  
  temp_data = processed_data[[n]]
  colnames(temp_data) = c("study_id", "phrase")

  #Create raw summary counts
  summarized_data = summarize_phrases(temp_data)
  summarized_data = summarized_data[!(is.na(summarized_data[,1])), ] 
  
  #Group similar words/phrases into a single count item.
  combined_summarized_data = combine_phrases(summarized_data)
  
  summaries[[n]] = combined_summarized_data

}

# The names produced above can be long and difficult to read or visualize. Use 
# the following to produce more readable summaries. 
# Create a copy of summaries where only the first phrase of each combined string is taken
# Scale the counts to make them more readable
# Remove counts that only appear 1
summaries2 = map(summaries, function(df) {
  df %>%
    filter(count > 1) %>%
    mutate(
      phrase = str_split(phrase, "; ", simplify = TRUE)[, 1],
      count = count^0.3
    )
})

# Create a copy of summaries where the longest (most descriptive?) phrase of each 
# combined string is taken
# Scale the counts to make them more readable
# Remove counts that only appear 1
summaries3 = map(summaries, function(df) {
  df %>%
    filter(count > 1) %>%
    mutate(
      phrase = sapply(str_split(phrase, "; "), function(x) {
      x[which.max(nchar(x))]
    }),
      count = count^0.3
    )
})

#=============================================================================
#Visualizing with ggplot
#=============================================================================
#This will use walk to cycle through the list, summaries
# Function to create a bar plot for a summary
create_bar_plot = function(summary_df, column_name) {
  ggplot(summary_df, aes_string(x = "phrase", y = "count")) +
    geom_bar(stat = "identity") +
    coord_flip() + # Flip coordinates for better readability
    theme_minimal() +
    labs(title = paste("Frequency of Phrases in", column_name),
         x = "Phrase",
         y = "Count")
}

# Create and save bar plots for each summary
walk2(column_names, summaries2, function(col, summary_df) {
  plot = create_bar_plot(summary_df, col)
  ggsave(paste0("./../output/bar_plot_", col, ".png"), plot, width = 8, height = 6)
})


#=============================================================================
#Word cloud
#=============================================================================
# Function to create a word cloud for a summary
create_word_cloud = function(summary_df, column_name) {
  wordcloud2(summary_df, size = .3, color = 'random-light', backgroundColor = "black")
}

# Create and save word clouds for each summary
walk2(column_names, summaries2, function(col, summary_df) {
  # Wordcloud2 requires a proper filename for HTML
  html_filename = paste0("./../output/word_cloud_", col, ".html")
  
  # Save word cloud as HTML file without embedding all dependencies
  saveWidget(create_word_cloud(summary_df, col), html_filename, selfcontained = FALSE)
})
