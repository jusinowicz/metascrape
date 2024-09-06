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
library(rcrossref)

#For plots
library(ggplot2)
library(wordcloud2)
library(htmlwidgets)
library(pandoc)
library(igraph)
#=============================================================================
# Load Data
#=============================================================================
ap_dir = "./../output/abstract_parsing1.csv"

# Read the CSV file
data = read_csv(ap_dir)
journal_key = read.csv(file = "journal_type.csv")
journal_key = journal_key[,2:3]
colnames(journal_key)[1:2] = c("journal", "type")

#=============================================================================
# User input: 
# These are helpful things to change so I will put them up front.
#=============================================================================
# The words/phrases that you would like to filter and get summary stats on.
ex_words = c("crop", "metal", "pollut", "agri", "contam", "mine", "minin", 
              "remedi", "toxic", "farm")

#Words associated with Ag
# The words/phrases that you would like to filter and get summary stats on.
ag_words = c("crop", "agri", "farm")

#Words associated with disturbance
dis_words = c("metal", "pollut",  "contam", "mine", "minin", "remedi", "toxic")

# The word clouds can be very finicky. When certain words/phrases have very 
# large counts, they might not print. 
# Use the "scale" parameter to change the the font scale, which helps. 
# Use the "transform" parameter to scale the counts themselves, which helps. 
#For the first filter
scale_ex = 0.5

#For the second part, more words, more complicated
scale2 =0.3 
#Transform for second part.
transform = 0.3 
#=============================================================================
#Custom functions
#=============================================================================
# Create a function to generate binary columns for each exclusion word
add_exclusions = function(df, column_name, words) {
  # Create a copy of the data frame to avoid modifying the original
  df_copy = df
  
  # Add binary columns for each exclusion word
  for (word in words) {
    df_copy = df_copy %>%
      mutate(!!paste0("contains_", word) := as.integer(str_detect(.data[[column_name]], regex(word, ignore_case = TRUE))))
  }
  
  return(df_copy)
}

#=============================================================================
# Get journal names based on DOIs using rcrossref
get_journal_name = function(doi) {
  result = tryCatch({
    cr_works(dois = doi) %>%
      pluck("data") %>%
      select(container.title) %>%
      unlist() %>%
      as.character()
  }, error = function(e) {
    return(NA)  # Return NA if the DOI is not found or any error occurs
  })
  return(result)
}

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
study_id = names(data)[1]
column_names = names(data)[names(data) != study_id]
#=============================================================================
# If you would like to apply any filters based on keywords.  
# E.g., filter out agrictultural studies
# And then get a count of how many studies were filtered by each phrase type
#=============================================================================
# Apply the function to filter out rows where e.g. 'crop' (or words in ex_words) 
# appears in the LANDUSE column
#What words:
ex_pattern = paste(ex_words, collapse = "|")

#Which column: 
col_use = column_names[6] #6 is LANDUSE 

# The "strict" version matches words exactly. 
# The "wild" version uses regex to match ANY appearance of a word. 
#Strict
# filtered_data = data %>%
#     filter(!str_detect(.data[[col_use]], fixed(ex_pattern, ignore_case = TRUE)))
#Wild
# filtered_data = data %>%
#   filter(!str_detect(.data[[col_use]], regex(ex_pattern, ignore_case = TRUE)))

#Wild but keep NAs, since these might be most likely to represent ecological studies
filtered_data = data %>%
  filter(is.na(.data[[col_use]]) | !str_detect(.data[[col_use]], regex(ex_pattern, ignore_case = TRUE)))

#Add journal name using cr_works from the rcrossref library
filtered_data$journal = sapply(filtered_data$DOI, get_journal_name)
filtered_data = filtered_data %>% left_join(journal_key, by = "journal")

#Export as csv if you want
file_name = paste("./../output/filtered_abstracts1.csv", sep="")
write.csv(file = file_name,filtered_data)

# Apply the function to add columns for each exclusion word
dat_ex = NULL
dat_ex = add_exclusions(data, col_use, ex_words)
dat_ex = dat_ex[,c(col_use, colnames(dat_ex)[(length(column_names)+2):dim(dat_ex)[2] ]) ]

# Summarize counts for each exclusion word.
# Add a column for total remaining studies. 
exclusion_summary = dat_ex %>%
  summarise(across(starts_with("contains_"), ~ sum(.x, na.rm = TRUE)))%>%
  mutate(remaining = nrow(data) - rowSums(select(., starts_with("contains_")), na.rm = TRUE)) %>%
  pivot_longer(everything(), names_to = "exclusion", values_to = "count")

#The other major subject areas, e.g. Ag and Disturbance
filtered_data2 = data[!(data$DOI  %in% filtered_data$DOI),]

#Ag:
ag_pattern = paste(ag_words, collapse = "|")
filtered_data_ag = filtered_data2 %>%
  filter(!is.na(.data[[col_use]]) & str_detect(.data[[col_use]], regex(ag_pattern, ignore_case = TRUE)))

#Disturbance:
dis_pattern = paste(dis_words, collapse = "|")
filtered_data_dis = filtered_data2 %>%
  filter(!is.na(.data[[col_use]]) & str_detect(.data[[col_use]], regex(dis_pattern, ignore_case = TRUE)))

#=============================================================================
#Visualizing with ggplot
#=============================================================================
# Function to create a bar plot for the exclusion summary
create_bar_plot = function(summary_df) {
  ggplot(summary_df, aes(x = reorder(exclusion, count), y = count)) +
    geom_bar(stat = "identity", fill = "skyblue") +
    coord_flip() + # Flip coordinates for better readability
    theme_minimal() +
    labs(title = "Exclusion Summary",
         x = "Phrase",
         y = "Count") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

# Create and save the bar plot for exclusion summary
plot = create_bar_plot(exclusion_summary)
ggsave(paste0("./../output/exclusion_summary_bar_plot.png"), plot, width = 8, height = 6)
#=============================================================================
#Word cloud
#=============================================================================
# Function to create a word cloud for a summary
create_word_cloud = function(summary_df, column_name) {
  wordcloud2(summary_df, size = scale_ex, color = 'random-light', backgroundColor = "black")
}

# Create and save the word cloud for exclusion summary
html_filename = "./../output/exclusion_summary_word_cloud.html"
saveWidget(create_word_cloud(exclusion_summary), html_filename, selfcontained = FALSE)
#=============================================================================
# This section attempts to unpack all of the individual phrases, and then 
# provide some summaries based on the kinds of phrases/words that were found.
#=============================================================================
#Split each label column into a list, then unpack all of the phrases
#into new entries. 
#
#>>>>>>>> Decide whether to use the data or filtered_data here <<<<<<<<
#
#This will unpack each word/phrase and create a long list where each
#word/phrase is matched with the Study ID.
#1. Data
processed_data = map(column_names, function(col){
  list_and_unnest(data, col,study_id)
})

#2. Filtered data
processed_data = map(column_names, function(col){
  list_and_unnest(filtered_data, col,study_id)
})

#Put the correct names back on the columns
for(n in 1:length(column_names)){ 
  colnames(processed_data[[n]])[2] = column_names[n]
  
  #Export as long csv lists
  file_name = paste("./../output/", column_names[n], ".csv", sep="")
  write.csv(file = file_name,processed_data[[n]] )
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
      count = count^transform
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
      count = count^transform
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

# Create and save bar plots for each summary
# Usine suffix _ex for exclusion summaries: 
walk2(column_names, summaries2, function(col, summary_df) {
  plot = create_bar_plot(summary_df, col)
  ggsave(paste0("./../output/bar_plot_", col, "_ex.png"), plot, width = 8, height = 6)
})


#=============================================================================
#Word cloud
#=============================================================================
# Function to create a word cloud for a summary
create_word_cloud = function(summary_df, column_name) {
  wordcloud2(summary_df, size = scale2, color = 'random-light', backgroundColor = "black")
}

# Create and save word clouds for each summary
walk2(column_names, summaries2, function(col, summary_df) {
  # Wordcloud2 requires a proper filename for HTML
  html_filename = paste0("./../output/word_cloud_", col, ".html")
  
  # Save word cloud as HTML file without embedding all dependencies
  saveWidget(create_word_cloud(summary_df, col), html_filename, selfcontained = FALSE)
})

# Create and save word clouds for each summary
# Usine suffix _ex for exclusion summaries:
walk2(column_names, summaries2, function(col, summary_df) {
  # Wordcloud2 requires a proper filename for HTML
  html_filename = paste0("./../output/word_cloud_", col, "_ex.html")
  
  # Save word cloud as HTML file without embedding all dependencies
  saveWidget(create_word_cloud(summary_df, col), html_filename, selfcontained = FALSE)
})


#=============================================================================
#Sunburst chart to summarize stats over high-level metadata
#=============================================================================
# Prepare the data for the sunburst chart
#####Level 1 data: 
#Total number of papers. 
#n_pap = dim(data)[1]

n_pap = sum(bio_sum$n)+ag_sum$n+dis_sum$n
#####Level 3 data: 
#For the Biological papers:
bio_sum = filtered_data %>% count(type)

#For the ag papers:
ag_sum =data.frame(type = "ag", n = dim(filtered_data_ag)[1])

#For the disturbance papers:
dis_sum =data.frame(type = "dis", n = dim(filtered_data_dis)[1])

# Prepare the sunburst data
sunburst_data = data.frame(
  labels = c(
    "Total Papers",  # Root node
    "Biological", "Agricultural", "Disturbance",  # First sublevel
    bio_sum$type,   # Second sublevel: Biological categories
    ag_sum$type, # Second sublevel: Agricultural categories
    dis_sum$type   # Second sublevel: Disturbance categories
  ),
  parents = c(
    "",  # Root node has no parent
    "Total Papers", "Total Papers", "Total Papers",  # First sublevel, all have "Total Papers" as parent
    rep("Biological", nrow( bio_sum)),  # Biological subcategories
    rep("Agricultural", nrow( ag_sum)),  # Agricultural subcategories
    rep("Disturbance", nrow( dis_sum))  # Disturbance subcategories
  ),
  values = c(
    n_pap,  # Total papers
    sum(bio_sum$n), sum(ag_sum$n), sum(dis_sum$n),  # Values for first sublevel (Biological, Agricultural, Disturbance)
    bio_sum$n,  # Values for second sublevel: Biological
    ag_sum$n,  # Values for second sublevel: Agricultural
    dis_sum$n  # Values for second sublevel: Disturbance
  )
)

# Replace NA labels (if necessary) with meaningful text or remove them
sunburst_data$labels[is.na(sunburst_data$labels)] <- "NA"

# Create the sunburst chart
sunburst_plot = plot_ly(
  sunburst_data,
  labels = ~labels,      # Use the 'labels' column for node names
  parents = ~parents,    # Use the 'parents' column for hierarchy
  values = ~values,      # Use the 'values' column for size (optional)
  type = 'sunburst',     # Specify this as a sunburst chart
  branchvalues = 'total',
  textinfo = 'label+value',  # Show label, percent, and value
  insidetextorientation = 'radial'  # Orientation of text
) %>%
  layout(title = "Paper breakdown")

# Display the plot
sunburst_plot
saveWidget(sunburst_plot, "./../output/paper_summary.html", selfcontained = FALSE)

#=============================================================================
#Directed trees to represent the stats over high-level data
#=============================================================================

# Create an edge list defining the connections
edges = c("Root", "Biological",
           "Root", "Agriculture",
           "Root", "Disturbance",
           "Biological", "SubBranch1",
           "Branch1", "SubBranch2",
           "Branch2", "SubBranch3",
           "Branch2", "SubBranch4")

# Define the numeric values for the edges
edge_weights <- c(1.5, 2.3, 0.8, 1.0, 2.5, 3.1)

# Create the graph
g <- graph(edges, directed = TRUE)

# Plot the tree with edge labels
plot(g, 
     vertex.size = 30, 
     vertex.label.cex = 1.5, 
     edge.arrow.size = 0.5,
     edge.label = edge_weights,  # Add numeric values as edge labels
     edge.label.cex = 1.2        # Control the size of the edge labels
)
