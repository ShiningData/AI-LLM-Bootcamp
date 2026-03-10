## Overview
Automated SQL Query Generation & Execution - Converts natural language questions into validated SQL queries against a PostgreSQL database
  (Chinook music store dataset), eliminating the need for manual SQL writing and reducing query errors through automated validation.

  Key business benefits:
  - Self-service analytics - Non-technical users can query databases directly
  - Error reduction - Built-in query validation and correction pipeline
  - Faster insights - Immediate answers to business questions like "Which genre has longest tracks?"
  - Reduced technical dependency - Business users don't need SQL expertise or database access

  The example demonstrates querying a music store database to analyze genre performance, which could inform inventory, marketing, or content
  curation decisions.

## Based on the Chinook music store database, here are business questions you can ask:

  Sales & Revenue Analysis

  - "Which artists generate the most revenue?"
  - "What are the top selling albums by total sales?"
  - "Which countries have the highest average order value?"
  - "What's our monthly sales trend?"

  Customer Analytics

  - "Who are our top spending customers?"
  - "Which cities have the most customers?"
  - "What's the average customer lifetime value?"
  - "Which customers haven't purchased in the last 6 months?"

  Product Performance

  - "Which music genres are most popular by sales volume?"
  - "What's the average track length by genre?"
  - "Which media types (MP3, AAC, etc.) sell best?"
  - "What are the longest and shortest tracks in our catalog?"

  Employee Performance

  - "Which sales employees have the highest revenue?"
  - "How many customers does each employee support?"
  - "What's the average sales per employee by territory?"

  Inventory & Catalog

  - "How many tracks does each artist have in our catalog?"
  - "Which playlists contain the most tracks?"
  - "What's the distribution of tracks across genres?"

  Try asking: "Which genre on average has the longest tracks?" (already in the code) or any of the above questions!

## Postgresql database
```bash
docker run --rm -d --name postgresql -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=Ankara06 -e POSTGRES_DB=traindb -p 5432:5432 postgres:15
```

## Copy scriipt to postgresql container
```bash
docker cp chinook_pg_serial_pk_proper_naming.sql postgresql:/
```

## Execute statement
```bash
docker exec -it postgresql bash
psql -d traindb -U postgres -f chinook_pg_serial_pk_proper_naming.sql
```

## Run app
```bash
python main.py
```