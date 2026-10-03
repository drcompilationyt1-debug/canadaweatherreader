# Historical news (optional)

Drop CSV files here to give the sentiment / LLM-news signals a *history* (otherwise those blocks
are only available live).  Any CSV with these columns works (case-insensitive):

| column                      | notes                                   |
|-----------------------------|-----------------------------------------|
| `ticker` / `stock` / `symbol` | e.g. `AAPL`                           |
| `date` / `datetime`         | any pandas-parsable timestamp           |
| `title` / `headline` / `text` | the headline                          |
| `summary` (optional)        | longer text                             |

Free datasets that drop straight in: Kaggle "Daily Financial News for 6000+ Stocks"
(`analyst_ratings_processed.csv`, columns `title, date, stock`) and "Financial News Headlines".

To let the LLM score history (costs tokens; capped by `signals.news_llm.max_history_calls`):

    stockbot build-dataset --set signals.news_llm.score_history=true
