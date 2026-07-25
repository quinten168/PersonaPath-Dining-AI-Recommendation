# Schema: `business_profiles`

Contains the aggregated and engineered features at the business level.

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `business_id` | STRING | ID of the business |
| `name` | STRING | Name of the business |
| `avg_vader_sentiment` | FLOAT | Average VADER compound score across all reviews (-1 to 1) |
| `dominant_vibe_tag` | STRING | The topic/vibe with the highest aggregate score for this business |
| `aggregated_topic_vector` | ARRAY<FLOAT> | The average topic distribution across all reviews for this business |
| `review_count` | INTEGER | Number of reviews after filtering |
