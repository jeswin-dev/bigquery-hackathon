-- Backfill embeddings for all rows (one-shot)
UPDATE `product_dataset.products`
SET embedding = (
  SELECT ml_generate_embedding_result
  FROM ML.GENERATE_EMBEDDING(
    MODEL `fit-aleph-471516-s1.product_dataset.your_embedding_model`,
    (SELECT CONCAT(IFNULL(title, ''), ' ', IFNULL(brand, '')) as content)
  )
)
WHERE embedding IS NULL OR ARRAY_LENGTH(embedding) = 0;  -- Only rows missing embeddings


-- Create remote embedding model (Vertex AI)
CREATE OR REPLACE MODEL product_dataset.your_embedding_model
REMOTE WITH CONNECTION  `fit-aleph-471516-s1.us.bigquery-vertex-connection`
OPTIONS(endpoint='gemini-embedding-001');

-- Batch backfill (process a limited slice each run)
UPDATE `product_dataset.products`
SET embedding = (
  SELECT ml_generate_embedding_result
  FROM ML.GENERATE_EMBEDDING(
    MODEL `fit-aleph-471516-s1.product_dataset.your_embedding_model`,
    (SELECT CONCAT(IFNULL(title, ''), ' ', IFNULL(brand, '')) as content)
  )
)
WHERE product_id IN (
  SELECT product_id 
  FROM `product_dataset.products`
  WHERE (embedding IS NULL OR ARRAY_LENGTH(embedding) = 0)
  ORDER BY product_id
  LIMIT 500
);

-- Simple similarity search (top 10 by cosine distance)
WITH search_embedding AS (
  SELECT ml_generate_embedding_result as embedding
  FROM ML.GENERATE_EMBEDDING(
    MODEL `fit-aleph-471516-s1.product_dataset.your_embedding_model`,
    (SELECT 'ceramic sink' as content)  -- Your search query
  )
)
SELECT 
  p.product_id,
  p.title,
  p.brand,
  ML.DISTANCE(s.embedding, p.embedding, 'COSINE') as cosine_distance,
  (1 - ML.DISTANCE(s.embedding, p.embedding, 'COSINE')) as similarity_score
FROM `product_dataset.products` p
CROSS JOIN search_embedding s
WHERE p.embedding IS NOT NULL 
  AND ARRAY_LENGTH(p.embedding) > 0
ORDER BY cosine_distance ASC  -- Most similar first
LIMIT 10;

-- Create remote LLM model (Vertex AI) for final selection
CREATE OR REPLACE MODEL `fit-aleph-471516-s1.product_dataset.your_llm_model`
REMOTE WITH CONNECTION `fit-aleph-471516-s1.us.bigquery-vertex-connection`
OPTIONS (
  ENDPOINT = 'gemini-2.0-flash'  -- or gemini-pro
);

-- Retrieve candidates, then ask LLM to pick one
WITH search_embedding AS (
  SELECT ml_generate_embedding_result as embedding
  FROM ML.GENERATE_EMBEDDING(
    MODEL `fit-aleph-471516-s1.product_dataset.your_embedding_model`,
    (SELECT 'ceramic sink' as content)  -- Your search query
  )
),
top_candidates AS (
  SELECT 
    p.product_id,
    p.title,
    p.brand,
    p.description,
    p.price,
    ML.DISTANCE(s.embedding, p.embedding, 'COSINE') as cosine_distance,
    (1 - ML.DISTANCE(s.embedding, p.embedding, 'COSINE')) as similarity_score
  FROM `product_dataset.products` p
  CROSS JOIN search_embedding s
  WHERE p.embedding IS NOT NULL 
    AND ARRAY_LENGTH(p.embedding) > 0
  ORDER BY cosine_distance ASC  -- Most similar first
  LIMIT 10
)
SELECT 
  -- Show all candidates found
  STRING_AGG(
    CONCAT(
      'Product ID: ', product_id, '\n',
      'Title: ', title, '\n',
      'Brand: ', brand, '\n',
      'Price: $', CAST(IFNULL(price, 0) AS STRING), '\n',
      'Similarity Score: ', CAST(ROUND(similarity_score, 3) AS STRING)
    ),
    '\n\n---\n\n'
  ) as found_products,
  
  -- LLM selection using a separate subquery
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
          'REASONING: [2-3 sentences explaining why this is the best choice]\n',
          'CONFIDENCE: [High/Medium/Low]\n',
          'KEY_FEATURES: [main features that make this the best match]\n\n',
          'Choose exactly ONE product and be decisive.'
        ) as prompt
        FROM top_candidates tc
      )
    )
  ) as selected_product

FROM top_candidates;

