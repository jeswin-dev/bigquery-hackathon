#!/usr/bin/env python3
import argparse
import os
import sys
import json
from typing import Optional

from google.cloud import bigquery


SQL_TEMPLATE = r'''
WITH search_embedding AS (
  SELECT ml_generate_embedding_result AS embedding
  FROM ML.GENERATE_EMBEDDING(
    MODEL `{project_id}.{dataset}.{embedding_model}`,
    (SELECT @search_query AS content)
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
  FROM `{project_id}.{dataset}.products` p
  CROSS JOIN search_embedding s
  WHERE p.embedding IS NOT NULL 
    AND ARRAY_LENGTH(p.embedding) > 0
  ORDER BY cosine_distance ASC
  LIMIT 10
)

SELECT 
  STRING_AGG(
    CONCAT(
      'Product ID: ', CAST(product_id AS STRING), '\n',
      'Title: ', IFNULL(title, ''), '\n',
      'Brand: ', IFNULL(brand, ''), '\n',
      'Price: $', CAST(IFNULL(price, 0) AS STRING), '\n',
      'Similarity Score: ', CAST(ROUND(similarity_score, 3) AS STRING)
    ),
    '\n\n---\n\n'
  ) AS found_products,
  (
    SELECT TO_JSON_STRING(ml_generate_text_result)
    FROM ML.GENERATE_TEXT(
      MODEL `{project_id}.{dataset}.{llm_model}`,
      (
        SELECT CONCAT(
          'SEARCH QUERY: "', @search_query, '"\n\n',
          'TASK: From the following 10 similar products, select the SINGLE best match for "', @search_query, '".\n\n',
          'CANDIDATES:\n',
          STRING_AGG(
            CONCAT(
              'Product ID: ', CAST(tc.product_id AS STRING), '\n',
              'Title: ', IFNULL(tc.title, ''), '\n',
              'Brand: ', IFNULL(tc.brand, ''), '\n',
              'Price: $', CAST(IFNULL(tc.price, 0) AS STRING), '\n',
              'Description: ', IFNULL(SUBSTR(tc.description, 1, 200), 'No description available'), '\n',
              'Similarity Score: ', CAST(ROUND(tc.similarity_score, 3) AS STRING)
            ),
            '\n\n---\n\n'
          ), '\n\n',
          'SELECTION CRITERIA:\n',
          '- Best relevance to "', @search_query, '"\n',
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
'''


def _extract_text_from_possible_json(s: str) -> str:
    if s is None:
        return ""
    t = s.strip()
    if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
        try:
            loaded = json.loads(t)
            if isinstance(loaded, str):
                return loaded
        except Exception:
            pass
    if t.startswith('{') or t.startswith('['):
        try:
            obj = json.loads(t)
            if isinstance(obj, str):
                return obj
            if isinstance(obj, dict):
                for key in ("ml_generate_text_result", "text", "content", "output_text", "result"):
                    val = obj.get(key)
                    if isinstance(val, str):
                        return val
                candidates = obj.get("candidates")
                if isinstance(candidates, list) and candidates:
                    texts: list[str] = []
                    for cand in candidates:
                        if not isinstance(cand, dict):
                            continue
                        # Direct text on candidate
                        for ckey in ("text", "output_text"):
                            cval = cand.get(ckey)
                            if isinstance(cval, str):
                                texts.append(cval)
                        # Content with parts
                        content = cand.get("content")
                        if isinstance(content, dict):
                            parts = content.get("parts")
                            if isinstance(parts, list):
                                for part in parts:
                                    if isinstance(part, dict):
                                        pt = part.get("text")
                                        if isinstance(pt, str):
                                            texts.append(pt)
                        for alt in ("ml_generate_text_result",):
                            aval = cand.get(alt)
                            if isinstance(aval, str):
                                texts.append(aval)
                    if texts:
                        return "\n".join(texts)
            if isinstance(obj, list) and obj:
                first = obj[0]
                if isinstance(first, str):
                    return first
                if isinstance(first, dict):
                    for key in ("content", "text", "output_text", "ml_generate_text_result"):
                        val = first.get(key)
                        if isinstance(val, str):
                            return val
        except Exception:
            pass
    return s


def parse_selected_product_id(text: str) -> Optional[str]:
    if not text:
        return None
    for line in text.splitlines():
        if not line:
            continue
        stripped = line.strip()
        if stripped.lower().startswith("selected_product_id:"):
            value = stripped.split(":", 1)[1].strip()
            if (value.startswith("[") and value.endswith("]")) or (value.startswith("(") and value.endswith(")")):
                value = value[1:-1].strip()
            if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                value = value[1:-1].strip()
            if value.endswith(","):
                value = value[:-1].strip()
            return value or None
    return None



def fetch_selected_product(
    client: bigquery.Client,
    project_id: str,
    dataset: str,
    search_query: str,
    embedding_model: str,
    llm_model: str,
):
    sql = SQL_TEMPLATE.format(
        project_id=project_id,
        dataset=dataset,
        embedding_model=embedding_model,
        llm_model=llm_model,
    )
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("search_query", "STRING", search_query),
        ]
    )
    result_iter = client.query(sql, job_config=job_config).result()
    row = next(iter(result_iter), None)
    if row is None:
        raise RuntimeError("Query returned no rows")

    found_products = row.get("found_products")
    selected_text = row.get("selected_product")

    selected_text = _extract_text_from_possible_json(selected_text or "")

    selected_id = parse_selected_product_id(selected_text or "")
    if not selected_id:
        raise RuntimeError(
            "Could not parse SELECTED_PRODUCT_ID from LLM output. Full text: "
            + (selected_text or "<empty>")
        )

    details_sql = f'''
        SELECT product_id, title, brand, price, description
        FROM `{project_id}.{dataset}.products`
        WHERE CAST(product_id AS STRING) = @pid
        LIMIT 1
    '''
    details_job = client.query(
        details_sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("pid", "STRING", selected_id)]
        ),
    )
    details_row = next(iter(details_job.result()), None)

    return {
        "search_query": search_query,
        "found_products": found_products,
        "selected_raw": selected_text,
        "selected_product_id": selected_id,
        "selected_product": dict(details_row.items()) if details_row else None,
    }



def main():
    parser = argparse.ArgumentParser(description="Select best product using BigQuery embeddings + LLM")
    parser.add_argument("search", help="Search query, e.g., 'ceramic sink'")
    parser.add_argument("--project-id", default=os.getenv("BQ_PROJECT", "fit-aleph-471516-s1"))
    parser.add_argument("--dataset", default=os.getenv("BQ_DATASET", "product_dataset"))
    parser.add_argument("--embedding-model", default=os.getenv("BQ_EMBED_MODEL", "your_embedding_model"))
    parser.add_argument("--llm-model", default=os.getenv("BQ_LLM_MODEL", "your_llm_model"))

    args = parser.parse_args()

    client = bigquery.Client(project=args.project_id)

    try:
        result = fetch_selected_product(
            client=client,
            project_id=args.project_id,
            dataset=args.dataset,
            search_query=args.search,
            embedding_model=args.embedding_model,
            llm_model=args.llm_model,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print("Search:", result["search_query"]) 
    print()
    print("Found candidates:\n")
    print(result.get("found_products") or "<none>")
    print()
    print("LLM selection (raw):\n")
    print(result.get("selected_raw") or "<none>")
    print()
    print("Selected product ID:", result.get("selected_product_id") or "<none>")
    print()
    selected = result.get("selected_product")
    if selected:
        print("Selected product details:")
        for k, v in selected.items():
            print(f"- {k}: {v}")
    else:
        print("Selected product details: <not found>")


if __name__ == "__main__":
    main()
