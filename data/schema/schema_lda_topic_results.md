# Schema: `lda_topic_results`

Contains the output of the Gensim LDA topic modeling stage.

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `review_id` | STRING | Unique identifier for the review |
| `business_id` | STRING | ID of the business being reviewed |
| `user_id` | STRING | ID of the user writing the review |
| `dominant_topic_id` | INTEGER | The topic ID (0-24) with the highest probability score for this review |
| `topic_probabilities` | ARRAY<FLOAT> | An array of 25 float values representing the probability of each topic in this review |
