# Schema: `user_personas`

Contains user-level features and persona classifications.

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `user_id` | STRING | ID of the user |
| `k_means_persona_id` | INTEGER | Cluster assignment from K-Means (1 of 6 personas) |
| `persona_label` | STRING | Human-readable label for the assigned persona |
| `persona_confidence_score` | FLOAT | Confidence level of the persona assignment based on distance to cluster center |
| `intent_combo_scores` | MAP<STRING, FLOAT> | Scores mapped to common dining intent combinations |
