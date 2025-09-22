## Semantic Product Finder on BigQuery (Embeddings + LLM)

A lean prototype that uses BigQuery SQL to: 1) turn product text into embeddings, 2) retrieve the closest products for a user query, and 3) ask an LLM to make the final, single-best choice. All of this happens inside BigQuery using remote Vertex AI models, so you don’t have to shuttle data around or run separate services.

### Why this exists
Search and recommendations often rely on keywords or manually curated rules. That breaks down with messy product data and inconsistent titles. This project leans on two primitives directly inside BigQuery:
- **Embeddings** to capture meaning beyond keywords
- **LLM reasoning** to make one decisive pick among near-duplicates

The end result: simple SQL that finds the right product for a query like “ceramic sink,” even if titles and descriptions vary wildly.

---

## What’s here
- `queries.sql`: Base SQL for model creation, embedding backfill, semantic search, and LLM selection
- `kaggle_description.md`: Challenge brief that inspired the build and scope

---

## How it works (in plain terms)
1. **Create an embedding model** (remote) that uses `gemini-embedding-001`.
2. **Backfill embeddings** for products (e.g., from `title` and `brand`).
3. **Run a semantic search**: embed the user query, compute cosine distance to product vectors, and take the top candidates.
4. **Create an LLM model** (remote) that uses `gemini-2.0-flash`.
5. **Ask the LLM to choose one** item from the top candidates using a short, structured prompt.

Everything is standard BigQuery SQL with `ML.GENERATE_EMBEDDING` and `ML.GENERATE_TEXT` over remote models.

---

## Setup

### Prerequisites
- BigQuery enabled in your Google Cloud project
- A BigQuery-to-Vertex AI connection (example ID below)
- A dataset (example: `product_dataset`) and a table `products` with at least:
  - `product_id` (unique id)
  - `title` (STRING)
  - `brand` (STRING)
  - `description` (STRING, optional but helpful)
  - `price` (NUMERIC/FLOAT, optional)
  - `embedding` (ARRAY<FLOAT64>)

### Remote model connection
Use or create a connection in the same region as your BigQuery dataset, e.g. `us`:
- Example connection: `fit-aleph-471516-s1.us.bigquery-vertex-connection`

You’ll need permissions to create remote models in BigQuery and to call Vertex AI.

---

## Key SQL (adapt as needed)

### 1) Create the embedding model
```sql
CREATE OR REPLACE MODEL product_dataset.your_embedding_model
REMOTE WITH CONNECTION  `fit-aleph-471516-s1.us.bigquery-vertex-connection`
OPTIONS(endpoint='gemini-embedding-001');
```

### 2) Backfill embeddings for products
This example derives text from `title` and `brand`.
```sql
UPDATE `product_dataset.products`
SET embedding = (
  SELECT ml_generate_embedding_result
  FROM ML.GENERATE_EMBEDDING(
    MODEL `fit-aleph-471516-s1.product_dataset.your_embedding_model`,
    (SELECT CONCAT(IFNULL(title, ''), ' ', IFNULL(brand, '')) AS content)
  )
)
WHERE embedding IS NULL OR ARRAY_LENGTH(embedding) = 0;
```

For large tables, run in batches:
```sql
UPDATE `product_dataset.products`
SET embedding = (
  SELECT ml_generate_embedding_result
  FROM ML.GENERATE_EMBEDDING(
    MODEL `fit-aleph-471516-s1.product_dataset.your_embedding_model`,
    (SELECT CONCAT(IFNULL(title, ''), ' ', IFNULL(brand, '')) AS content)
  )
)
WHERE product_id IN (
  SELECT product_id
  FROM `product_dataset.products`
  WHERE (embedding IS NULL OR ARRAY_LENGTH(embedding) = 0)
  ORDER BY product_id
  LIMIT 500
);
```

### 3) Create the LLM model
```sql
CREATE OR REPLACE MODEL `fit-aleph-471516-s1.product_dataset.your_llm_model`
REMOTE WITH CONNECTION `fit-aleph-471516-s1.us.bigquery-vertex-connection`
OPTIONS (
  ENDPOINT = 'gemini-2.0-flash'
);
```

### 4) Search + LLM selection (single decisive pick)
This query does three things: embed the search text, select the top 10 nearest products, and ask the LLM to choose exactly one.
```sql
WITH search_embedding AS (
  SELECT ml_generate_embedding_result AS embedding
  FROM ML.GENERATE_EMBEDDING(
    MODEL `fit-aleph-471516-s1.product_dataset.your_embedding_model`,
    (SELECT 'ceramic sink' AS content)
  )
),

top_candidates AS (
  SELECT
    p.product_id,
    p.title,
    p.brand,
    p.description,
    p.price,
    ML.DISTANCE(s.embedding, p.embedding, 'COSINE') AS cosine_distance,
    (1 - ML.DISTANCE(s.embedding, p.embedding, 'COSINE')) AS similarity_score
  FROM `product_dataset.products` p
  CROSS JOIN search_embedding s
  WHERE p.embedding IS NOT NULL AND ARRAY_LENGTH(p.embedding) > 0
  ORDER BY cosine_distance ASC
  LIMIT 10
)

SELECT
  STRING_AGG(
    CONCAT(
      'Product ID: ', product_id, '\n',
      'Title: ', title, '\n',
      'Brand: ', brand, '\n',
      'Price: $', CAST(IFNULL(price, 0) AS STRING), '\n',
      'Similarity Score: ', CAST(ROUND(similarity_score, 3) AS STRING)
    ),
    '\n\n---\n\n'
  ) AS found_products,
  (
    SELECT ml_generate_text_result
    FROM ML.GENERATE_TEXT(
      MODEL `fit-aleph-471516-s1.product_dataset.your_llm_model`,
      (
        SELECT CONCAT(
          'SEARCH QUERY: "ceramic sink"\n\n',
          'TASK: From the following 10 similar products, select the SINGLE best match for "ceramic sink".\n\n',
          'CANDIDATES:\n',
          STRING_AGG(
            CONCAT(
              'Product ID: ', tc.product_id, '\n',
              'Title: ', tc.title, '\n',
              'Brand: ', tc.brand, '\n',
              'Price: $', CAST(IFNULL(tc.price, 0) AS STRING), '\n',
              'Description: ', IFNULL(SUBSTR(tc.description, 1, 200), 'No description available'), '\n',
              'Similarity Score: ', CAST(ROUND(tc.similarity_score, 3) AS STRING)
            ),
            '\n\n---\n\n'
          ), '\n\n',
          'SELECTION CRITERIA:\n',
          '- Best relevance to "ceramic sink"\n',
          '- Quality and features\n',
          '- Value for money\n',
          '- Brand reputation\n\n',
          'RESPONSE FORMAT:\n',
          'SELECTED_PRODUCT_ID: [exact product_id]\n',
          'PRODUCT_NAME: [title]\n',
          'REASONING: [2-3 sentences]\n',
          'CONFIDENCE: [High/Medium/Low]\n',
          'KEY_FEATURES: [main features]\n\n',
          'Choose exactly ONE product and be decisive.'
        ) AS prompt
        FROM top_candidates tc
      )
    )
  ) AS selected_product
FROM top_candidates;
```

To change the query, replace the literal `'ceramic sink'` in two places above.

---

## Notes, tuning, and costs
- **Indexing**: On very large tables (≈ 1M+ rows), create a vector index and use `VECTOR_SEARCH` for speed. This prototype uses brute-force distance for simplicity.
- **Prompt control**: The LLM prompt is intentionally short and structured. It’s easy to tighten or relax rules based on your domain.
- **Data quality**: Embeddings work best with clean, descriptive text. Consider concatenating title, brand, and a normalized description.
- **Costs**: You’ll incur BigQuery compute for UPDATEs/queries and Vertex AI model invocation costs for embeddings and text generation.
- **Permissions**: Model creation and remote calls require appropriate IAM roles in BigQuery and Vertex AI.

---

## Extending this prototype
- Add a second-pass re-rank based on business rules (price bands, stock, margin).
- Return top-N with LLM-generated rationales instead of a single pick.
- Log LLM selections and user clicks for evaluation and continuous prompt tuning.
- Swap in `AI.GENERATE` / `AI.GENERATE_TABLE` if you prefer the newer AI SQL APIs.

---

## Troubleshooting
- If `embedding` stays NULL, confirm the remote connection and that the model exists at `product_dataset.your_embedding_model`.
- If calls fail with permission errors, verify IAM for the BigQuery connection and Vertex AI.
- If similarity looks off, check that `embedding` is `ARRAY<FLOAT64>` and populated for all rows.

---

## Credits
Built as a focused demonstration for the BigQuery AI challenge brief. All SQL lives in one place and runs where the data lives.
